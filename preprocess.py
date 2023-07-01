import os.path as osp
import argparse

from torch_geometric.datasets import OGB_MAG
from torch_geometric.transforms import AddMetaPaths
from torch_geometric.utils import bipartite_subgraph
import torch

args = argparse.ArgumentParser()
args.add_argument('--data_path', type=str, default='./data')

def load_dataset(path):
    dataset = OGB_MAG(root=path, preprocess='metapath2vec')
    data = dataset[0]
    del data[('paper', 'cites', 'paper')]
    data['author'].nid = torch.arange(data['author'].x.size(0)).reshape(-1, 1)

    view_1 = [('paper', 'writes_by', 'author'), ('author', 'writes', 'paper')]  # ('paper', 'cites', 'paper')
    view_2_ = [('institution', 'affiliated_with_by', 'author'), ('author', 'affiliated_with', 'institution')]  # Initial graph's edge type
    view_3 = [('field_of_study', 'has_topic_by', 'paper'), ('paper', 'has_topic', 'field_of_study')]
    view_dict = [view_1, view_2_, view_3]
    edge_type, rev_edge_type = view_1[0], view_1[1]   # edge type of main view
    data.u_type, data.v_type = edge_type[0], edge_type[2]  # node types of main view

    for view in view_dict:
        data[view[0]].edge_index = torch.flipud(data.edge_index_dict[view[1]]) # directed -> undirected
    
    metapaths = [[view_1[0], view_2_[1]], [view_2_[0], view_1[1]]]
    data = AddMetaPaths(metapaths)(data)  # Add metapaths as auxiliary view
    del data[view_2_[0]]
    del data[view_2_[1]]  # Delete redundant edge type

    view_2 = [('institution', 'metapath_1', 'paper'), ('paper', 'metapath_0', 'institution')]  # Generate new edge type around paper
    view_dict = [view_1, view_2, view_3]

    train_data, val_data, test_data = get_data_split(data, edge_type, rev_edge_type, view_dict)
    print('Split dataset successfully !')  # Split dataset into train/val/test to satisfy the inductive setting

    return train_data, val_data, test_data, view_dict


def get_subgraph(data, mask, view_dict):
    u_type = data.u_type  
    
    for view in view_dict:
        for edge_type in view:
            if edge_type[0] == u_type:
                v_set = torch.arange(data.x_dict[edge_type[2]].size(0))  # 只对paper 节点进行分割，其他节点全部保留
                subset = (mask.nonzero().squeeze(), v_set)
            elif edge_type[2] == u_type:
                u_set = torch.arange(data.x_dict[edge_type[0]].size(0))
                subset = (u_set, mask.nonzero().squeeze())
            else:
                continue
            edge_index, _ = bipartite_subgraph(subset, data.edge_index_dict[edge_type], relabel_nodes=True)
            data[edge_type].edge_index = edge_index

    data[u_type].x = data.x_dict[u_type][mask]
    data[u_type].y = data[u_type].y[mask]
    del data[u_type].train_mask 
    del data[u_type].test_mask 
    del data[u_type].val_mask
    data[u_type].mask = torch.ones(len(data[u_type].x), dtype=torch.bool)   # reindex the mask
    return data


def get_data_split(data, edge_type, rev_edge_type, view_dict, num_train=0.8, num_val=0.1, num_test=0.1):
    u_type, v_type = data.u_type, data.v_type
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
 
    num_u = data[u_type].x.size(0)
    num_v = data[v_type].x.size(0)
    
    if not hasattr(data[u_type], 'train_mask'):
        print('Manual division ')
        num_train = int(num_train * num_u)
        num_val = int(num_val * num_u)
        
        perm = torch.randperm(num_u)
        train_mask = torch.zeros(num_u, dtype=torch.bool)
        val_mask = torch.zeros(num_u, dtype=torch.bool)
        test_mask = torch.zeros(num_u, dtype=torch.bool)
        train_mask[perm[:num_train]] = True       
        val_mask[perm[num_train:num_train+num_val]] = True
        test_mask[perm[num_train+num_val:]] = True

        train_data[u_type].train_mask = train_mask
        val_data[u_type].val_mask = val_mask
        test_data[u_type].test_mask = test_mask

    # Eliminate redundant edges to satisfy "inductive" and "strict cold start" scenarios, only for view 1.
    print(f'the num of train data edges: {train_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    train_data = get_subgraph(train_data, train_data[u_type].train_mask, view_dict)
    print(f'{train_data[edge_type].edge_index.size(-1)}')

    print(f'the num of val data edges: {val_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    val_data = get_subgraph(val_data, val_data[u_type].val_mask, view_dict)
    print(f'{val_data[edge_type].edge_index.size(-1)}')

    print(f'the num of test data edges: {test_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    test_data = get_subgraph(test_data, test_data[u_type].test_mask, view_dict)
    print(f'{test_data[edge_type].edge_index.size(-1)}')

    return train_data, val_data, test_data


if __name__ == "__main__":
    args = args.parse_args()

    train_data, val_data, test_data, view_dict = load_dataset(args.data_path)
    torch.save(train_data, osp.join(args.data_path, 'train_data.pt'))
    torch.save(val_data, osp.join(args.data_path, 'val_data.pt'))
    torch.save(test_data, osp.join(args.data_path, 'test_data.pt'))
    print('Save preprocess dataset successfully !')