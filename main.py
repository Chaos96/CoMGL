import gc
import json
import numpy as np
import os
import random
import torch

from base_options import BaseOptions
from utils import assign_free_gpus

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_idx)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main():
    args = BaseOptions().get_arguments()
    print(args)
    if torch.cuda.is_available(): 
        gpus_to_use = assign_free_gpus()
        args.cuda_idx = int(gpus_to_use[0])
    set_seed(args)
    args.device = torch.device(f'cuda:{args.cuda_idx}' if torch.cuda.is_available() else torch.device('cpu'))
        # args.device = device = torch.device('cpu')
    print(args.device)

    if args.exp_mode == 'link_prediction':
        from trainer_link_prediction import trainer
    elif args.exp_mode == 'node_prediction':
        from trainer_node_prediction import trainer
    
    trainer(args).main()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
    