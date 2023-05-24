import logging
import os
import sys
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models

#import nni

def set_env(seed, gpu):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--order', type=bool, default=False)
    parser.add_argument('--cases', type=str, default='one')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--gpu', type= int, default=0, help = 'set gpu number')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12, choices=[3, 6, 9, 12],help='the number of time interval for predcition, default as 12')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3) # choices=[3, 2]
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50, help='epochs, default as 50')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--early', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()
    
    if args.order == False:
        print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed, args.gpu)
    
    paths = f'./Results/cases_{args.cases}'
    
    try:
        os.mkdir('./Results')
        os.mkdir(paths)
    except: 
        pass
    
    arg_dict = vars(args)
    
    f = open(os.path.join(paths, 'parameters.txt'), 'w')
    
    for i,j in arg_dict.items(): 
        f.write(f'{i} = {j} \n')
    f.close()
        
    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks, paths

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    if args.order == True:

        len_val = int(math.floor(data_col * 0.1))
        len_train = int(math.floor(data_col * 0.7))
        
        # print(f"length of whole dataset: {data_col}")
        # print(f"length of train and validation: {len_train}, {len_val}")

        train, val = dataloader.load_data(args, len_train, len_val)
        zscore = preprocessing.StandardScaler()
        train = zscore.fit_transform(train)
        val = zscore.transform(val)
        
        x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
        x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
         
        train_data = utils.data.TensorDataset(x_train, y_train)
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
        val_data = utils.data.TensorDataset(x_val, y_val)
        val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
        
        return n_vertex, zscore, train_iter, val_iter
    
    elif args.order == False:
        train_and_test_rate = 0.8
        len_train = int(math.floor(data_col * train_and_test_rate))
        len_test = int(data_col - len_train)
        
        # print(f"length of whole dataset: {data_col}")
        # print(f"length of train and test dataset: {len_train}, {len_test}")

        
        train, test = dataloader.load_data(args, len_train, len_test)
        zscore = preprocessing.StandardScaler()
        train = zscore.fit_transform(train)
        test = zscore.transform(test)

        x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
        x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

        train_data = utils.data.TensorDataset(x_train, y_train)
        train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
        test_data = utils.data.TensorDataset(x_test, y_test)
        test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
        
        return n_vertex, zscore, train_iter, test_iter

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    tr_loss_list, val_loss_list = [], []
    
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        train_loss = l_sum / n
        tr_loss_list.append(train_loss)
        
        # if args.order == True:
        #     val_loss = val(model, val_iter)
             # val_loss_list.append(val_loss)
        
        if args.early == True:
            if es.step(val_loss):
                print('Early stopping.')
                break 
            
    if args.order == True:
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        # print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
        #     format(epoch+1, optimizer.param_groups[0]['lr'], train_loss, val_loss, gpu_mem_alloc))
        sys.exit(print(val_loss.item()))
    
    elif args.order == False:
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], train_loss, gpu_mem_alloc))
        
        return tr_loss_list

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    test_res = {"MSE":[test_MSE], "MAE": [test_MAE], "RMSE": [test_RMSE], "WMAPE": [test_WMAPE]}
    print(f'Dataset {args.dataset:s} | Test loss {test_res["MSE"][0]:.6f} | MAE {test_res["MAE"][0]:.6f} | RMSE {test_res["RMSE"][0]:.6f} | WMAPE {test_res["WMAPE"][0]:.8f}')
    
    return test_res

def save_result(paths, tr_loss_list, test_loss, args): # val_loss_list
    
    pd.DataFrame(tr_loss_list).to_csv(os.path.join(paths, 'train_loss.csv'), sep=',', index=False)
    pd.DataFrame(test_loss).to_csv(os.path.join(paths, 'test_loss.csv'), sep=',', index=False)
    
    loss_dict = {"train": tr_loss_list}
    
    epoch_range = np.arange(1, len(loss_dict["train"])+1)
        
    plt.plot(epoch_range, loss_dict["train"], label='train')
    plt.legend()
    plt.title(f'Loss per epoch', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel(f'Loss', fontweight='bold')
    plt.savefig(os.path.join(paths, f'{args.cases}_TrainValLoss.png'), dpi=500)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args, device, blocks, paths = get_parameters()
    
    if args.order == True:
        n_vertex, zscore, train_iter, val_iter= data_preparate(args, device)
        loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
        train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
    
    elif args.order == False:
        n_vertex, zscore, train_iter, test_iter= data_preparate(args, device)
        loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
        tr_loss_list = train(loss, args, optimizer, scheduler, es, model, train_iter, test_iter)
        test_res = test(zscore, loss, model, test_iter, args)
        save_result(paths, tr_loss_list, test_res, args)
        