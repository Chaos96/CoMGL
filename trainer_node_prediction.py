import argparse
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score

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
            batch_size = batch_data[u_type].batch_size       
            batch_data = batch_data.to(args.device)
            model.optimizer.zero_grad()

            # main view's negative sampling
            num_nodes = [batch_data.x_dict[u_type].size(0), batch_data.x_dict[u_type].size(0)]
            edge_label_index, edge_label = batch_get_neg_edges(batch_data[args.edge_type].edge_index, num_nodes=num_nodes)
            batch_data[args.edge_type].edge_label_index, batch_data[args.edge_type].edge_label = edge_label_index, edge_label

            # auxiliary views' negative sampling
            for idx, view in enumerate(args.view_dict[1:]):  
                edge_type = view[1]
                v_type = edge_type[2]
                num_nodes = [batch_data.x_dict[u_type].size(0), batch_data.x_dict[v_type].size(0)]
                batch_data[edge_type].edge_label_index, batch_data[edge_type].edge_label = batch_get_neg_edges(batch_data[edge_type].edge_index, num_nodes=num_nodes)        

            # 聚合auxiliary views中的paper embedding
            z, paper_emb = model.aggregator(model, batch_data, args.view_dict)
            author_emb = model.encoder[0](batch_data.x_dict, batch_data.edge_index_dict)[u_type]
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

            # candidate edges generate
            if args.generate_edges:
                edge_label_index, edge_label = batch_data[args.edge_type].edge_label_index, batch_data[args.edge_type].edge_label
                edge_label = F.one_hot(edge_label.long())
                logits = model.predictor[0](paper_emb, author_emb, edge_label_index).sigmoid()
                edge_loss = F.binary_cross_entropy_with_logits(logits, edge_label.to(logits))

            if args.dataset == 'mag':
                logits = model.predictor[-1](paper_emb)
                y = batch_data[u_type].y
            elif args.dataset == 'dblp':
                logits = model.predictor[-1](author_emb)
                y = batch_data[u_type].y

            label_loss = F.cross_entropy(logits[:batch_size], y[:batch_size].to(args.device))

            if args.generate_edges:
                loss = 0.5 * edge_loss_aux + edge_loss + label_loss
            loss.backward()

            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(args.para_list, args.grad_clip_norm)
        
            model.optimizer.step()

        return loss.item()

    def batch_test(self, data_loader):
        args = self.args   
        model = self.model
        self.model.encoder.eval()
        self.model.predictor.eval()

        u_type, v_type = args.u_type, args.v_type

        with torch.no_grad():
            acc = []
            for batch_data in data_loader:
                del batch_data[args.edge_type]   
                del batch_data[args.rev_edge_type]   # 测试阶段删除main view 边信息
                batch_size = batch_data[u_type].batch_size  
                batch_data = batch_data.to(args.device)

                # candidate edges set of main view
                num_nodes = [batch_data.x_dict[u_type].size(0), batch_data.x_dict[u_type].size(0)]
                num_edges = 1000
                edge_index_candidate = get_candidate_edges(num_nodes=num_nodes, num_edges=num_edges)    # main view 

                # auxiliary views negative sampling
                for idx, view in enumerate(args.view_dict[1:]):  
                    edge_type = view[1]
                    v_type = edge_type[2]
                    num_nodes = [batch_data.x_dict[u_type].size(0), batch_data.x_dict[v_type].size(0)]
                    batch_data[edge_type].edge_label_index, batch_data[edge_type].edge_label = batch_get_neg_edges(batch_data[edge_type].edge_index, num_nodes=num_nodes)       

                z, paper_emb = model.aggregator(model, batch_data, args.view_dict)
                author_nid = batch_data[v_type].nid.squeeze()
                author_emb = self.author_emb[author_nid].to(args.device)

                if args.generate_edges:
                    logits = model.predictor[0](paper_emb, author_emb, edge_index_candidate)
                    logits = F.gumbel_softmax(logits, tau=0.01, hard=True)
                    edge_index = edge_index_candidate.to(args.device) * logits[:, 1].long()
                    batch_data[args.edge_type].edge_index = edge_index
                    batch_data[args.rev_edge_type].edge_index = torch.flipud(edge_index)

                x_dict_ = batch_data.x_dict
                x_dict_[u_type], x_dict_[v_type] = paper_emb.to(args.device), author_emb.to(args.device)
                x_dict_ = model.encoder[0](x_dict_, batch_data.edge_index_dict)
                if args.dataset == 'mag':
                    y_pred = model.predictor[-1](x_dict_[u_type]).softmax(dim=1).detach().cpu()
                    y = batch_data[u_type].y.cpu()
                elif args.dataset == 'dblp':
                    y_pred = model.predictor[-1](x_dict_[v_type]).softmax(dim=1).detach().cpu()
                    y = batch_data[v_type].y.cpu()
                acc.append(accuracy_score(y[:batch_size], y_pred.argmax(dim=1)[:batch_size]))
        acc = np.array(acc)
        return acc.mean()

    def main(self):
        args = self.args   
        args.device = device = torch.device(f'cuda:{args.cuda_idx}' if torch.cuda.is_available() else torch.device('cpu'))
        # args.device = device = torch.device('cpu')
        print(device)

        if args.wandb:
            wandb.init(project='MultiView')
            wandb.run.name = 'multiview_sage' + time.strftime("-%b-%d-%H:%M", time.localtime())
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
                print(loss)
                val_acc = self.batch_test(val_loader)
                test_acc = self.batch_test(test_loader)
                print(f'epoch: {epoch}, loss: {loss:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
