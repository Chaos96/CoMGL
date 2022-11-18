import os
import time
import subprocess

import numpy as np
from torch_geometric.datasets import DBLP, OGB_MAG, HGBDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import (AddMetaPaths, RandomLinkSplit,
                                        RandomNodeSplit)
from torch_geometric.utils import bipartite_subgraph
from ogb.nodeproppred import Evaluator
import torch


def assign_free_gpus(threshold_vram_usage=10000, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = np.array([
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ])  # Remove garbage
        rank = np.argsort(gpu_info)
        # Keep gpus under threshold only
        free_gpus = [i for i in rank if gpu_info[i] < threshold_vram_usage]

        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join([str(i) for i in free_gpus])
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
    return gpus_to_use

def load_dataset(args):
    if args.dataset == 'mag':
        dataset = OGB_MAG(root=args.data_path, preprocess='metapath2vec')
        data = dataset[0]
        del data[('paper', 'cites', 'paper')]
        args.node_class_num = data['paper'].y.unique().max() + 1  # 349

        data['author'].nid = torch.arange(data['author'].x.size(0)).reshape(-1, 1)

        view_1 = [('paper', 'writes_by', 'author'), ('author', 'writes', 'paper')]  # ('paper', 'cites', 'paper')
        view_2_ = [('institution', 'affiliated_with_by', 'author'), ('author', 'affiliated_with', 'institution')]  # Initial graph's edge type
        view_3 = [('field_of_study', 'has_topic_by', 'paper'), ('paper', 'has_topic', 'field_of_study')]
        view_dict = [view_1, view_2_, view_3]
        for view in view_dict:
            data[view[0]].edge_index = torch.flipud(data.edge_index_dict[view[1]]) # directed -> undirected
        
        if os.path.exists('./data/mag_processed.pt'):
            print('load data successfully !')
            data = torch.load('../data/mag_processed.pt')
        else:
            metapaths = [[view_1[0], view_2_[1]], [view_2_[0], view_1[1]]]
            data = AddMetaPaths(metapaths)(data)
            del data[view_2_[0]]
            del data[view_2_[1]]
            torch.save(data, './data/mag_processed.pt')

        view_2 = [('institution', 'metapath_1', 'paper'), ('paper', 'metapath_0', 'institution')]  # Generate new edge type around paper
        args.view_dict = view_dict = [view_1, view_2, view_3]
        author_emb = data['author'].x.clone()    # record authors'embedding during training for inductive settings

    elif args.dataset == 'dblp':
        dataset = DBLP(root=args.data_path)
        data = dataset[0]
        args.node_class_num = data['author'].y.unique().max() + 1  # 4

        data["conference"].x = torch.ones(data["conference"].num_nodes, 1)

        view_1 = [('paper', 'to', 'author'), ('author', 'to', 'paper')]
        view_2 = [('paper', 'to', 'conference'), ('conference', 'to', 'paper')]
        view_3 = [('paper', 'to', 'term'), ('term', 'to', 'paper')]
        view_dict = view_dict = [view_1, view_2, view_3]

    # main view's edge_type
    args.edge_type = edge_type = view_1[0]
    args.rev_edge_type = rev_edge_type = view_1[1]   
    args.metadata = data.metadata()

    train_data, val_data, test_data = get_data_split(data, edge_type, rev_edge_type, view_dict)

    train_data = add_delete_edges(train_data, view_dict=view_dict, noise_ratio=args.noise_ratio)

    if args.train_on_subgraph:
        train_loader = NeighborLoader(
            train_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [15] * 2 for key in train_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=1024 * 5,
            input_nodes=('paper', train_data['paper'].train_mask),
        )

        mask = torch.ones(val_data['paper'].x.size(0), dtype=torch.bool)
        val_loader = NeighborLoader(
            val_data,
            num_neighbors={key: [15] * 2 for key in val_data.edge_types},
            batch_size=1024,
            input_nodes=('paper', mask),
        )

        mask = torch.ones(test_data['paper'].x.size(0), dtype=torch.bool)
        test_loader = NeighborLoader(
            test_data,
            num_neighbors={key: [15] * 2 for key in test_data.edge_types},
            batch_size=1024,
            input_nodes=('paper', mask),
        )
        return train_loader, val_loader, test_loader, author_emb

    return train_data, val_data, test_data, author_emb

def filter_edges(edge_index, node_mask, loc=0):
    node_set = node_mask.nonzero()
    mask = torch.isin(edge_index[loc], node_set)
    return edge_index[:, mask]

def negative_sampling(edge_index, u_set, v_set, ratio=1):
    size = int(edge_index.size(-1) * ratio)
    row = np.random.choice(u_set, size=size).reshape(1, -1)
    col = np.random.choice(v_set, size=size).reshape(1, -1)
    neg_edge_index = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0).to(edge_index)
    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(edge_index.size(-1)), torch.zeros(neg_edge_index.size(-1))], dim=0)
    return edge_label_index, edge_label

def add_delete_edges(data, view_dict, noise_ratio):
    for view in view_dict:
        for edge_type in view:
            edge_index = data.edge_index_dict[edge_type]
            num_edges = edge_index.size(-1)
            if noise_ratio <= 0:   # delete edges
                noise_ratio = np.abs(noise_ratio)
                perm = torch.randperm(num_edges)[:int(num_edges * noise_ratio)]
                data.edge_index_dict[edge_type] = edge_index[:, perm]
            
            else:
                loc = 0 if edge_type[0]=='paper' else 1
                paper_set = torch.unique(edge_index[loc]).numpy()
                v_type = edge_type[2] if loc==0 else edge_type[0]
                v_set = np.arange(data.x_dict[v_type].size(0))
                edge_index, _ = negative_sampling(edge_index, paper_set, v_set, ratio=noise_ratio)
                data.edge_index_dict[edge_type] = edge_index

    return data

def get_subgraph(data, mask, view_dict):
    data['paper'].x = data.x_dict['paper'][mask]
    for view in view_dict:
        for edge_type in view:
            if edge_type[0] == 'paper':
                v_mask = torch.ones(data.x_dict[edge_type[2]].size(0), dtype=torch.bool)
                subset = (mask, v_mask)
            else:
                u_mask = torch.ones(data.x_dict[edge_type[0]].size(0), dtype=torch.bool)
                subset = (u_mask, mask)
            edge_index, _ = bipartite_subgraph(subset, data.edge_index_dict[edge_type], relabel_nodes=True)
            data[edge_type].edge_index = edge_index
    
    return data

def get_data_split(data, edge_type, rev_edge_type, view_dict, num_train=0.8, num_val=0.1, num_test=0.1):
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
 
    num_paper = data['paper'].x.size(0)
    num_author = data['author'].x.size(0)
    
    if not hasattr(data['paper'], 'train_mask'):
        print('Manual division ')
        num_train = int(num_train * num_paper)
        num_val = int(num_val * num_paper)
        
        perm = torch.randperm(num_paper)
        train_mask = torch.zeros(num_paper, dtype=torch.bool)
        val_mask = torch.zeros(num_paper, dtype=torch.bool)
        test_mask = torch.zeros(num_paper, dtype=torch.bool)
        train_mask[perm[:num_train]] = True       
        val_mask[perm[num_train:num_train+num_val]] = True
        test_mask[perm[num_train+num_val:]] = True

        train_data['paper'].train_mask = train_mask
        val_data['paper'].val_mask = val_mask
        test_data['paper'].test_mask = test_mask

    # Eliminate redundant edges to satisfy "inductive" and "strict cold start" scenarios, only for view 1.
    print(f'the num of train data edges: {train_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    train_data[edge_type].edge_index = filter_edges(train_data[edge_type].edge_index, train_data['paper'].train_mask, 0)
    train_data[rev_edge_type].edge_index = filter_edges(train_data[rev_edge_type].edge_index, train_data['paper'].train_mask, 1)
    # train_data = get_subgraph(train_data, train_data['paper'].train_mask, view_dict[1:])
    print(f'{train_data[edge_type].edge_index.size(-1)}')


    val_data = get_subgraph(val_data, val_data['paper'].val_mask, view_dict)

    test_data = get_subgraph(test_data, test_data['paper'].test_mask, view_dict)

    return train_data, val_data, test_data