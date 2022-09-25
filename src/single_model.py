import argparse
from audioop import rms
import logging
import os
import sys
from itertools import product
from time import localtime, sleep, strftime, time

import numpy as np
import setproctitle # to set the name of process
import torch
import torch.utils
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp # 多线程工作

from dataset import get_data_queue_cf, get_data_queue_cf_nonsparse, get_data_queue_efficiently, get_data_queue_negsampling_efficiently, get_data_queue_subsampling_efficiently
from models import (CML, DELF, DMF, FISM, GMF, MLP, SVD, JNCF_Cat, JNCF_Dot, SVD_plus_plus, SPACE, BaseModel, Virtue_CF)
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from train_eval import (evaluate_cf, evaluate_cf_efficiently, evaluate_cf_efficiently_implicit, get_arch_performance_cf_signal_param_device, get_arch_performance_single_device, train_single_cf, train_single_cf_efficiently,get_arch_performance_implicit_single_device)

import GPUtil
import socket
import math


def single_model_run(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    setproctitle.setproctitle(args.process_name) # 设定进程名称
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    log_format = '%(asctime)s %(message)s' # 记录精确的实践
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w', format=log_format, datefmt='%m/%d %I:%M:%S %p')
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    # args.save = 'save/'
    save_name = args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + '_' + args.opt + str(args.lr)
    save_name += '_' + str(args.data_type)

    if args.mode == 'reinforce':
        save_name += '_' + str(args.controller_lr) + '_' + args.controller + '_' + str(args.controller_batch_size)
    else:
        # save_name += '_' + str(args.weight_decay) # default=1e-5
        save_name += '_' + ('%.6f' % (args.weight_decay)) 
    save_name += '_' + str(args.seed) # default=1
    save_name += '_' + current_time
    # save_name = args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + '_' 
    # + args.opt + str(args.lr) + '_' + str(args.weight_decay) + '_' 
    # + str(args.seed) + '_' + current_time

    # 创建log路径
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.save + '/log'):
        os.makedirs(args.save + '/log')
    if not os.path.exists(args.save + '/log_sub'):
        os.makedirs(args.save + '/log_sub')
    if os.path.exists(os.path.join(args.save, save_name + '.txt')):
        os.remove(os.path.join(args.save, save_name + '.txt'))
    
    # fh表示
    fh = logging.FileHandler(os.path.join(args.save + 'log', save_name + '.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    writer = SummaryWriter(log_dir=args.save + 'tensorboard/{}'.format(save_name))
    if args.use_gpu: # default = True
        torch.cuda.set_device(args.gpu)
        logging.info('gpu device = %d' % args.gpu)
    else:
        logging.info('no gpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_start = time()
    dim = 2
    data_path = args.dataset + '/'

    # setting datasets,  default='ml-100k'
    if args.dataset == 'ml-100k': # default
        num_users = 943
        num_items = 1682
    elif args.dataset == 'ml-1m':
        num_users = 6040
        num_items = 3952
    elif args.dataset == 'ml-10m':
        num_users = 71567
        num_items = 65133
    elif args.dataset == 'ml-20m':
        num_users = 138493
        num_items = 131262
    elif args.dataset == 'youtube_small':
        num_ps = 600
        num_qs = 14340
        num_rs = 5
        dim = 3
    elif args.dataset == 'youtube':
        num_ps = 15088
        num_qs = 15088
        num_rs = 5
        dim = 3
    elif args.dataset == 'amazon-book':
        num_users = 11899
        num_items = 16196
    elif args.dataset == 'yelp':
        num_users = 26829
        num_items = 20344
    elif args.dataset == 'yelp2':
        num_users = 15496
        num_items = 12666
    else:
        pass
    args.num_users = num_users
    args.num_items = num_items

    if args.data_type == 'implicit': # 主要使用这一行，隐式推荐
        # train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
        train_queue_pair, valid_queue, test_queue = get_data_queue_subsampling_efficiently(data_path, args)
        
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
    # print(train_queue)
    logging.info('prepare data finish! [%f]' % (time()-data_start))
    stored_arches = {} # log ging表示添加到记录中


    # 分不同的mode运行代码  default='random_single', help='search or single mode'
    # print("args.mode: {}".format(args.mode)) # 应该只要运行一次即可
    print('Debug in model {}'.format(args.mode))
    
    # 大概是explicit的情况下才使用
    # 不搜索，采用单独的模型进行分析
    if args.mode == 'DELF' or 'SVD_plus_plus' or 'FISM' or 'SVD' or 'GMF' or 'MLP' or 'CML' or 'JNCF_Dot' or 'JNCF_Cat' or 'DMF':
        # 根据参数选取对应的一个模型
        start = time()
        sleep(1)
        if args.mode == 'DELF': # implicit不行都是sparese tensor,explicit
            model = DELF(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'SVD_plus_plus': #implicit, explicit不行
            model = SVD_plus_plus(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'FISM':
            model = FISM(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'SVD':
            model = SVD(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'GMF':
            model = GMF(num_users, num_items, args.embedding_dim, args.weight_decay) # 第一阶段用这个
        elif args.mode == 'MLP':
            model = MLP(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'CML':
            model = CML(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'JNCF_Dot':
            model = JNCF_Dot(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'JNCF_Cat':# implicit，explicit有问题，size
            model = JNCF_Cat(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'DMF': # implicit,explicit有问题  sparse tensors do not have strides
            model = DMF(num_users, num_items, args.embedding_dim, args.weight_decay)
        elif args.mode == 'MF': 
            model = Virtue_CF(num_users, num_items, args.embedding_dim, args.weight_decay)
        else:
            print('other mode...')
            pass
        if args.use_gpu:
            model = model.cuda()
        if args.mode == 'test_mlp':
            optimizer = torch.optim.Adagrad(model.train_parameters(), args.lr) # 默认Adagrad优化， 可以进行分析
        elif args.opt == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr)
        else:
            optimizer = torch.optim.Adagrad(model.parameters(), args.lr)# default
        losses = []
        performances = []

        # 对explicit和implicit两种模式进行分裂 (yan modified)
        if args.data_type == 'explicit': # 组装数据
            if args.use_gpu:
                train_queue = [k.cuda() for k in train_queue] # 对单个模型这里容易出问题，train_queue未定义
            train_queue[3] = train_queue[3].to_sparse() # 转换为SparseDataFrame
            train_queue[4] = train_queue[4].to_sparse()
            if args.use_gpu:
                test_queue = [k.cuda() for k in test_queue]
            test_queue[3] = test_queue[3].to_sparse()
            test_queue[4] = test_queue[4].to_sparse()
        elif args.data_type == 'implicit':
            if args.use_gpu:
                train_queue_pair = [k.cuda() for k in train_queue_pair]
            train_queue_pair[3] = train_queue_pair[3].to_sparse()
            train_queue_pair[4] = train_queue_pair[4].to_sparse()
        else:
            pass

        all_users = torch.tensor(list(range(num_users)), dtype=torch.int64).repeat_interleave(num_items)
        all_items = torch.tensor(list(range(num_items)), dtype=torch.int64).repeat(num_users)
        with torch.cuda.device(args.device):
            all_users = all_users.cuda()
            all_items = all_items.cuda()

        rmse_list = []
        loss_list = []
        for train_epoch in range(args.train_epochs):
            # train
            if args.data_type == 'explicit': # 用不到
                loss = train_single_cf_efficiently(train_queue, model, optimizer, args)
            else: # 只用implicit：隐式
                loss = train_single_cf_efficiently(train_queue_pair, model, optimizer, args)
            losses.append(loss)
            if 'test' not in args.mode:
                if train_epoch > 1000:
                    down = 4096 if args.minibatch else train_queue[0].shape[0]
            if dim == 2: # default 2， 验证集合上的操作
                if 'DMF' in args.mode or 'JNCF' in args.mode:
                    if args.data_type == 'explicit':
                        rmse = evaluate_cf_efficiently(model, test_queue, sparse='ui')
                    else:
                        rmse = evaluate_cf_efficiently_implicit(model, test_queue, all_users, all_items, args)
                else: # 使用GMF,MLP等
                    if args.data_type == 'explicit':
                        rmse = evaluate_cf_efficiently(model, test_queue, sparse='')
                    else:
                        rmse  = evaluate_cf_efficiently_implicit(model, test_queue, all_users, all_items, args) # debugging，实际返回值是recall < 1
            performances.append(rmse)#eval
            if args.data_type == 'explicit':
                logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (train_epoch, loss, rmse, time()-start))
            elif args.data_type == 'implicit':
                logging.info('train_epoch: %d, loss: %.4f, recall20: %.4f[%.4f]' % (train_epoch, loss, rmse, time()-start))
            else:
                pass
            rmse_list.append(rmse)
            loss_list.append(loss)
            
            writer.add_scalar('train/loss', loss, train_epoch) #tensorboard
            writer.add_scalar('train/rmse', rmse, train_epoch)
        # end of for training

    else:
        print('No such mode.')
    print("save_name: {}".format(save_name))
    # recall20_array = np.array(rmse_list)
    # max_epoch  = np.argmax(recall20_array)
    # max_recall = np.max(recall20_array)
    return rmse_list, loss_list
