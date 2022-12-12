import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score
from ogb.linkproppred import Evaluator

from comgl.model import *
from comgl.utils import *
from comgl.loss import *
from utils import *
from evaluate import Evaluator


model = MLPPredictor(2, 128, 128, )

class trainer(): 
    def __init__(self, args):
        self.args = args    
        
    def batch_train(self, data_loader):
        args = self.args
        model = self.model
        model.train()
        
        loss_record = []
        for batch_data in tqdm(data_loader, leave=False):         
            batch_size = batch_data[args.u_type].batch_size    
            batch_data = batch_data.to(args.device)
            model.optimizer.zero_grad()

            batch_data = global_negative_sampling(args, batch_data, ratio=[1, 0.2])

            z, u_emb = model.aggregator(model, batch_data, args.view_dict)
            v_emb = batch_data.x_dict[args.v_type]
            # v_emb = model.encoder[0](batch_data.x_dict, batch_data.edge_index_dict)[args.v_type]
            # v_nid = batch_data[args.v_type].nid.squeeze()
            # self.global_v_emb[v_nid] = v_emb.detach().cpu()  # 记录训练阶段的v emb

            # auxiliary views' construction loss
            edge_loss_aux = 0
            for idx, view in enumerate(args.view_dict[1:]): 
                if args.u_type == 'paper':
                    edge_type = view[1]
                else:
                    edge_type = view[0]
                v_type = edge_type[2]
                split = batch_data[edge_type].split
                pos_edge, neg_edge = split['pos_edge'], split['neg_edge']
                num_neg = neg_edge.size(-1)
                pos_out = model.predictor[idx+1](z[idx][args.u_type], z[idx][v_type], pos_edge.to(args.device))
                neg_out = model.predictor[idx+1](z[idx][args.u_type], z[idx][v_type], neg_edge.to(args.device))
                edge_loss_aux += self.calculate_loss(pos_out, neg_out, num_neg)

            split = batch_data[args.edge_type].split
            pos_edge, neg_edge = split['pos_edge'], split['neg_edge']
            num_neg = neg_edge.size(-1)
            pos_out = model.predictor[0](u_emb, v_emb, pos_edge.to(args.device))
            neg_out = model.predictor[0](u_emb, v_emb, neg_edge.to(args.device))
            edge_loss = self.calculate_loss(pos_out, neg_out, num_neg)
            loss = 0.5 * edge_loss_aux + edge_loss
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.para_list, args.grad_clip_norm)
        
            model.optimizer.step()

            loss_record.append(loss.item())
    
        return np.mean(loss_record)

    def test(self, data_loader, evaluator):
        args = self.args   
        model = self.model
        model.eval()

        u_type, v_type = args.u_type, args.v_type
        with torch.no_grad():
            for batch_data in tqdm(data_loader, leave=False):  
                batch_size = batch_data[args.u_type].batch_size  
                batch_data = batch_data.to(args.device)

                batch_data = global_negative_sampling(args, batch_data, ratio=[1, 0.2])

                z, u_emb = model.aggregator(model, batch_data, args.view_dict)
                v_emb = batch_data.x_dict[v_type]
                # v_nid = batch_data[args.v_type].nid.squeeze()
                # v_emb = self.global_v_emb[v_nid].to(args.device)

                split = batch_data[args.edge_type].split
                pos_edge, neg_edge = split['pos_edge'], split['neg_edge']
                pos_pred = model.predictor[0](u_emb, v_emb, pos_edge.to(args.device))
                neg_pred = model.predictor[0](u_emb, v_emb, neg_edge.to(args.device))

        eval_metric = args.eval_metric
        if eval_metric == 'hits':
            results = evaluate_hits(
                evaluator,
                pos_pred,
                neg_pred)
        
        elif eval_metric == 'mrr':
            results = evaluate_mrr(
                evaluator,
                pos_pred,
                neg_pred)

        elif 'recall_my' in eval_metric:
            results = evaluate_recall_my(
                evaluator,
                pos_pred,
                neg_pred, topk=eval_metric.split('@')[1])
        else:
            results = evaluate_rocauc(
                evaluator,
                pos_pred,
                neg_pred)
        return results
    
    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        loss_func_name = self.args.loss_func_name
        if loss_func_name == 'ce_loss':
            loss = ce_loss(pos_out, neg_out)
        elif loss_func_name == 'info_nce_loss':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif loss_func_name == 'log_rank_loss':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif loss_func_name == 'adaptive_auc_loss' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def main(self):
        args = self.args   

        if args.wandb:
            wandb.init(project='CoMGL')
            wandb.run.name = '' + time.strftime("-%b-%d-%H:%M", time.localtime())
            wandb.config.update(args)

        if args.train_on_subgraph:
            train_loader, val_loader, test_loader, self.global_v_emb = load_dataset(args)

        model = CoMGL(
            args,
            view_dict=args.view_dict,
            predictor_name=args.predictor_name,
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            aggregator_name=args.aggregator_name,
            mlp_num_layers=args.mlp_num_layers,
            gnn_hidden_channels=args.gnn_hidden_channels,
            agg_hidden_channels=args.agg_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,  
            optimizer_name=args.optimizer,
            device=args.device
        )

        self.model = model


        # total_params = sum(p.numel() for param in model.u for p in param)
        # print(f'Total number of model parameters: {total_params}')

        evaluator = Evaluator(name=args.dataset, eval_metric='rocauc')

        for run in range(args.runs):
            # model.param_init()
            start_time = time.time()
            for epoch in range(1, 1 + args.epochs):
                print(f'Train: ')
                loss = self.batch_train(train_loader)
                print(f'Val: ')
                val_results = self.test(val_loader, evaluator)
                print(f'Test: ')
                test_results = self.test(test_loader, evaluator)
                if args.eval_metric == 'ROC-AUC':
                    print(f"epoch: {epoch}, loss: {loss:.4f}, val: {val_results['rocauc']:.3f}, test: {test_results['rocauc']:.3f}\n")
