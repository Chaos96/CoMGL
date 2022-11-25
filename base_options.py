import argparse
import numpy as np
import os
import torch

class BaseOptions():
    def get_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_mode', type=str, default='link_prediction', choices=['link_prediction', 
        'node_prediction'])
        parser.add_argument('--dataset', type=str, default='openid', choices=['dblp', 'mag', 'openid'])
        parser.add_argument('--data_path', type=str, default='~/public_data/pyg_data/')
        parser.add_argument('--auxiliary_view_num', type=int, default=2)
        parser.add_argument('--train_on_subgraph', default=True, action="store_false")
        parser.add_argument('--generate_edges', default=False, action="store_true")
        parser.add_argument('--use_view_1', default=False, action="store_true")

        # training hyperparameter
        parser.add_argument('--cuda', default=True, action="store_false")  
        parser.add_argument('--runs', type=int, default=1)  
        parser.add_argument('--epochs', type=int, default=150)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=64 * 1024)    
        parser.add_argument('--noise_ratio', type=float, default=0.2)     # auxiliary views' info noise ratio      
        parser.add_argument('--eval_steps', type=int, default=1)
        parser.add_argument('--eval_metric', type=str, default='ROC-AUC', choices=['ROC-AUC', 'hits', 'mrr', 'recall_my@0.8', 'recall_my@1', 'recall_my@1.25', 'recall_my@0'])
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--embed_size', type=int, default=128)
        parser.add_argument('--predictor_name', type=str, default='MLP')
        parser.add_argument('--gnn_num_layers', type=int, default=2)
        parser.add_argument('--aggregator_name', type=str, default='Uncertainty', choices=['Uncertainty', 'Attention', 'Add'])
        parser.add_argument('--mlp_num_layers', type=int, default=2)
        parser.add_argument('--gnn_hidden_channels', type=int, default=128)
        parser.add_argument('--agg_hidden_channels', type=int, default=128)
        parser.add_argument('--mlp_hidden_channels', type=int, default=128)
        parser.add_argument('--grad_clip_norm', type=float, default=2.0)
        parser.add_argument('--loss_func', type=str, default='BCE')
        parser.add_argument('--optimizer', type=str, default='Adam')       
        parser.add_argument('--patience', type=int, default=20)

        parser.add_argument('--wandb', action='store_true', help="whether to open wandb")

        args = parser.parse_args()

        return args


        