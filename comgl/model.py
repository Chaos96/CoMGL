import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.nn as pygnn
from .layer import *
from .utils import *

def calculate_weight(weight, y_pred, edge_index, loc=1):
    from torch_scatter import scatter_mean
    idx = edge_index[loc, :].max() + 1
    weight[:idx] = scatter_mean(y_pred, edge_index[loc, :])
    return weight

class GNNEncoder(torch.nn.Module):
    def __init__(self, view, metadata, hidden_channels, num_layers, dropout):
        super().__init__()

        self.convs = nn.ModuleList()
        self.dropout = dropout
        for layer in range(num_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv(-1, hidden_channels) for edge_type in metadata[1] if edge_type in view}
            )
            self.convs.append(conv)

        self.norm = nn.ModuleList([pygnn.norm.BatchNorm(hidden_channels) for node_type in metadata[0]])

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.norm[idx](x) for idx, (key, x) in enumerate(x_dict.items())}
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return x_dict


class CoMGL(nn.Module):
    def __init__(self, args, view_dict, predictor_name, lr, dropout, grad_clip_norm, gnn_num_layers, aggregator_name, mlp_num_layers, 
                gnn_hidden_channels, agg_hidden_channels, mlp_hidden_channels, optimizer_name, device):
        super(CoMGL, self).__init__()

        self.args = args
        self.clip_norm = grad_clip_norm
        self.device = device

        self.edge_type = self.args.edge_type

        self.encoder = nn.ModuleList()   # 每个view 单独建立一个GNNEncoder
        for idx, view in enumerate(view_dict):
            self.encoder.append(GNNEncoder(view, args.metadata, 
                                    hidden_channels=gnn_hidden_channels,
                                    num_layers=gnn_num_layers,
                                    dropout=dropout))
        self.aggregator = create_aggregator_layer(aggregator_name=args.aggregator_name,
                                                hidden_channels=args.agg_hidden_channels,
                                                auxiliary_view_num=args.auxiliary_view_num,
                                                use_view1=args.use_view_1)

        self.predictor = create_predictor_layer(input_channels=gnn_hidden_channels,
                                                hidden_channels=mlp_hidden_channels,
                                                out_channels=[2, args.node_class_num],
                                                num_layers=mlp_num_layers,
                                                dropout=dropout,
                                                predictor_name=predictor_name)
        
        self.para_list = list(self.encoder.parameters()) + list(self.aggregator.parameters()) + list(self.predictor.parameters())
        args.para_list = self.para_list

        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)
        
        self.encoder, self.aggregator, self.predictor = self.encoder.to(device), self.aggregator.to(device), self.predictor.to(device)
    
    def param_init(self):
        self.encoder.reset_parameters()
        self.aggregator.reset_parameters()
        self.predictor.reset_parameters()
    
    def encode_aux(self, x_dict, edge_index_dict, auxiliary_view_num=2):
        z = []
        for i in range(auxiliary_view_num):
            z.append(self.encoder[i+1](x_dict, edge_index_dict))
        return z


def create_aggregator_layer(hidden_channels, aggregator_name='Uncertainty', auxiliary_view_num=2, use_view1=False):
    # aggregator = nn.ModuleList()
    aggregator_name = aggregator_name.upper()
    if aggregator_name == 'UNCERTAINTY':
        return Uncertrainty_estimate()
    elif aggregator_name == 'ATTENTION':
        return SelfAttention(hidden_channels, hidden_channels, hidden_channels)


def create_predictor_layer(input_channels, hidden_channels, out_channels, num_layers, dropout=0., predictor_name='MLP', auxiliary_task_num=2):
    predictor = nn.ModuleList()
    predictor_name = predictor_name.upper()
    if predictor_name == 'MLP':
        predictor.append(MLPPredictor(num_layers, input_channels, hidden_channels, out_channels[0], dropout)) # main task: edge predict
        if auxiliary_task_num:          
            for _ in range(auxiliary_task_num):  # auxiliary task: edge predict
                predictor.append(MLPPredictor(num_layers, input_channels, hidden_channels, out_channels[0], dropout))

        predictor.append(nn.Sequential(     # main view: node prediction layer
          nn.Linear(hidden_channels, hidden_channels),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(hidden_channels, out_channels[-1]),
          nn.Softmax(dim=1)
        ))
        return predictor