import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score

from comgl.model import *
from comgl.utils import *
from utils import *



class trainer():
    def __init__(self, args):
        self.args = args    
        
    def batch_train(self, data_loader):
        args = self.args
        model = self.model
        model.encoder.train()
        model.aggregator.train()
        model.predictor.train()

        u_type, v_type = args.u_type, args.v_type
        
        for batch_data in data_loader:         
            batch_size = batch_data[args.u_type].batch_size    
            batch_data = batch_data.to(args.device)
            model.optimizer.zero_grad()

            # negative sampling
            num_nodes = [batch_data.x_dict[args.u_type].size(0), batch_data.x_dict[args.v_type].size(0)]
            batch_data[args.edge_type].edge_label_index, batch_data[args.edge_type].edge_label = batch_get_neg_edges(batch_data[args.edge_type].edge_index, num_nodes=num_nodes)    # main view 

            # auxiliary views negative sampling
            for idx, view in enumerate(args.view_dict[1:]): 
                edge_type = view[1]
                v_type = edge_type[2]
                num_nodes = [batch_data.x_dict[args.u_type].size(0), batch_data.x_dict[v_type].size(0)]
                batch_data[edge_type].edge_label_index, batch_data[edge_type].edge_label = batch_get_neg_edges(batch_data[edge_type].edge_index, num_nodes=num_nodes)  

            z, paper_emb = self.model.aggregator(self.model, batch_data, args.view_dict)
            author_emb = model.encoder[0](batch_data.x_dict, batch_data.edge_index_dict)[args.v_type]
            author_nid = batch_data[args.v_type].nid.squeeze()
            self.author_emb[author_nid] = author_emb.detach().cpu()  # 记录训练阶段的author emb

            # auxiliary views' construction loss
            edge_loss_aux = 0
            for idx, view in enumerate(args.view_dict[1:]): 
                edge_type = view[1]
                v_type = edge_type[2]
                edge_label_index_aux, edge_label_aux = batch_data[edge_type].edge_label_index, batch_data[edge_type].edge_label
                edge_label_aux = F.one_hot(edge_label_aux.long(), num_classes=2)
                logits = model.predictor[idx+1](z[idx][u_type], z[idx][v_type], edge_label_index_aux.to(args.device))
                edge_loss_aux += F.binary_cross_entropy_with_logits(logits, edge_label_aux.to(logits))

            edge_label_index, edge_label = batch_data[args.edge_type].edge_label_index, batch_data[args.edge_type].edge_label
            logits = self.model.predictor[0](paper_emb, author_emb, edge_label_index)
            edge_label = F.one_hot(edge_label.long(), num_classes=2)
            edge_loss = F.binary_cross_entropy_with_logits(logits[:batch_size], edge_label.to(logits)[:batch_size])
            loss = 0.5 * edge_loss_aux + edge_loss
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.encoder.parameters(), args.grad_clip_norm)
                nn.utils.clip_grad_norm_(self.model.aggregator.parameters(), args.grad_clip_norm)
                nn.utils.clip_grad_norm_(self.model.predictor.parameters(), args.grad_clip_norm)
        
            self.model.optimizer.step()
    
        return loss.item()

    def test(self, data_loader):
        args = self.args   
        model = self.model
        self.model.encoder.eval()
        self.model.predictor.eval()

        for batch_data in data_loader:
            batch_size = batch_data[args.u_type].batch_size  
            batch_data = batch_data.to(args.device)

            num_nodes = [batch_data.x_dict[args.u_type].size(-1), batch_data.x_dict[args.v_type].size(-1)]
            edge_label_index, edge_label = batch_get_neg_edges(batch_data[args.edge_type].edge_index[:, :1000], num_nodes=num_nodes)
            for idx, view in enumerate(args.view_dict[1:]):  # auxiliary views
                edge_type = view[1]
                v_type = edge_type[2]
                num_nodes = [batch_data.x_dict[args.u_type].size(0), batch_data.x_dict[v_type].size(0)]
                batch_data[edge_type].edge_label_index, batch_data[edge_type].edge_label = batch_get_neg_edges(batch_data[edge_type].edge_index, num_nodes=num_nodes)

            z, paper_emb = self.model.aggregator(self.model, batch_data, args.view_dict)
            author_nid = batch_data[args.v_type].nid.squeeze()
            author_emb = self.author_emb[author_nid].to(args.device)

            logits = self.model.predictor[0](paper_emb, author_emb, edge_label_index).sigmoid()
            y = edge_label.cpu().numpy()
            y_pred = logits.squeeze().cpu().detach().numpy()

        return roc_auc_score(y, y_pred), average_precision_score(y, y_pred)

    def main(self):
        args = self.args   
        args.device = device = torch.device(f'cuda:{args.cuda_idx}' if torch.cuda.is_available() else torch.device('cpu'))
        # args.device = device = torch.device('cpu')
        print(device)

        if args.wandb:
            wandb.init(project='CoMGL')
            wandb.run.name = '' + time.strftime("-%b-%d-%H:%M", time.localtime())
            wandb.config.update(args)

        if args.train_on_subgraph:
            train_loader, val_loader, test_loader, self.author_emb = load_dataset(args)

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
            device=device
        )

        self.model = model


        # total_params = sum(p.numel() for param in model.para_list for p in param)
        # print(f'Total number of model parameters: {total_params}')

        for run in range(args.runs):
            # model.param_init()
            start_time = time.time()
            for epoch in range(1, 1 + args.epochs):
                loss = self.batch_train(train_loader)
                val_auc, val_ap = self.test(val_loader)
                auc, ap = self.test(test_loader)
                print(f'epoch: {epoch}, loss: {loss:.4f}, val_auc: {val_auc:.4f}, val_ap: {val_ap:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
