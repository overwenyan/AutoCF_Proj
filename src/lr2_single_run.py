import numpy as np
# import matplotlib.pyplot as plt
import os
from utils import *
# import datetime
import argparse
# from audioop import rms
import os
# import sys
import numpy as np
import json
from torch import multiprocessing as mp # 多线程工作
from single_model import single_model_run

parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=0.05, help='learning rate for arch encoding')
parser.add_argument('--controller_lr', type=float, default=1e-1, help='learning rate for controller')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--update_freq', type=int, default=1, help='frequency of updating architeture')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--use_gpu', type=int, default=1, help='whether use gpu')
parser.add_argument('--minibatch', type=int, default=1, help='whether use minibatch')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--train_epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--search_epochs', type=int, default=1000, help='num of searching epochs')
parser.add_argument('--save', type=str, default='save/', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='random_single', help='search or single mode')
parser.add_argument('--process_name', type=str, default='AutoCF@wenyan', help='process name')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
parser.add_argument('--controller', type=str, default='PURE', help='structure of controller')
parser.add_argument('--controller_batch_size', type=int, default=4, help='batch size for updating controller')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--max_batch', type=int, default=65536, help='max batch during training')
parser.add_argument('--device', type=int, default=0, help='GPU device')
parser.add_argument('--multi', type=int, default=0, help='using multi-training for single architecture')
parser.add_argument('--if_valid', type=int, default=1, help='use validation set for tuning single architecture or not')
parser.add_argument('--breakpoint', type=str, default='save/log.txt', help='the log file storing existing results')
parser.add_argument('--arch_file', type=str, default='src/arch.txt', help='all arches')
parser.add_argument('--remaining_arches', type=str, default='src/arch.txt', help='')
parser.add_argument('--arch_assign', type=str, default='[0,3]', help='')
parser.add_argument('--data_type', type=str, default='implicit', help='explicit or implicit(default)')
parser.add_argument('--loss_func', type=str, default='bprloss', help='Implicit loss function')
parser.add_argument('--mark', type=str, default='') # 

args = parser.parse_args()
mp.set_start_method('spawn', force=True) # 一种多任务运行方法

if __name__ == '__main__':
    logfilePath = './lr_random/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    
    # for lr test random run
    anchor_config_num = 3
    # lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 128+1, 1), anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-4, high=1e2, size=anchor_config_num)

    gpu_num = args.gpu
    # args.gpu = gpu_num
    args.device = gpu_num
    # args.mode = 'GMF'
    args.mode = 'MLP'
    
    args.save = logfilePath
    # args.dataset = 'ml-1m'
    args.dataset = 'ml-100k'
    args.data_type = 'implicit'

    lr_hp_list = ['1e-6','1e-5', '1e-4','1e-3', '1e-2', '1e-1', '1e0','1e1','1e2', '1e3', '1e4', '1e5']
    lr_hp_list = [str(float(lr)) for lr in lr_hp_list]
    lr_result_list_dict = {}
    for lr in lr_hp_list:
        lr_result_list_dict[lr] = []
    
    # best_scores = np.zeros((3, anchor_config_num))
    for i in range(anchor_config_num):
        for lr in lr_hp_list:
            args.lr = float(lr)
            args.weight_decay = weight_decay_list[i]
            args.embedding_dim = embedding_dim_list[i]
            args.opt = opt_list[i]
            rmse_list, loss_list = single_model_run(args)
            recall20_array = np.array(rmse_list)
            # max_epoch  = np.argmax(recall20_array)
            # max_recall = np.max(recall20_array)
            
            max_recall = np.mean(recall20_array[-50:])
            lr_result_list_dict[lr].append(max_recall)
            print("[Index: {}|{}],  mean_recall: {}".format(i+1,anchor_config_num, max_recall))
            # print("[Index: {}|{}], max_epoch: {},  max_recall: {}".format(i+1,anchor_config_num, max_epoch, max_recall))
            # os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} --lr {} --embedding_dim {} --weight_decay {:.4f}'.format(dataset, gpu_num, gpu_num, logfilePath, opt_list[i], float(lr), embedding_dim_list[i], weight_decay_list[i]))
                    
    print('lr_result_list_dict: {}'.format(lr_result_list_dict))

    # dumps 将数据转换成字符串
    info_json = json.dumps(lr_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    f = open(logfilePath+'lrinfo'+'148'+'.json', 'w')
    # f = open(logfilePath+'lrinfo21.json', 'w')
    f.write(info_json)