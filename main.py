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

from dataset import get_data_queue_cf, get_data_queue_cf_nonsparse, get_data_queue_efficiently, get_data_queue_negsampling_efficiently
from models import (CML, DELF, DMF, FISM, GMF, MLP, SVD, JNCF_Cat, JNCF_Dot, SVD_plus_plus, SPACE)
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from train_eval import (evaluate_cf, evaluate_cf_efficiently, evaluate_cf_efficiently_implicit, get_arch_performance_cf_signal_param_device, get_arch_performance_single_device, train_single_cf, train_single_cf_efficiently,get_arch_performance_implicit_single_device)

import GPUtil
import socket
import math

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
parser.add_argument('--save', type=str, default='logs', help='experiment name')
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

# 一些辅助函数
def get_hyperparam_performance(x):
    arch, num_users, num_items, train_queue, test_queue, args, param, device_id = x

    return get_arch_performance_cf_signal_param_device(arch, num_users, num_items, train_queue, test_queue, args, param, device_id)

def get_single_model_performance(x):
    if len(x) == 10:
        arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device_id, if_valid = x
        return get_arch_performance_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device_id, if_valid)
    else:
        arch, num_users, num_items, train_queue, valid_queue, test_queue, train_queue_pair, args, param, device_id, if_valid = x
        return get_arch_performance_implicit_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, train_queue_pair, args, param, device_id, if_valid)


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    setproctitle.setproctitle(args.process_name) # 设定进程名称
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    log_format = '%(asctime)s %(message)s' # 记录精确的实践
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w', format=log_format, datefmt='%m/%d %I:%M:%S %p')
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    args.save = 'save/'
    save_name = args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + '_' + args.opt + str(args.lr)
    save_name += '_' + str(args.data_type)

    if args.mode == 'reinforce':
        save_name += '_' + str(args.controller_lr) + '_' + args.controller + '_' + str(args.controller_batch_size)
    else:
        save_name += '_' + str(args.weight_decay) # default=1e-5
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
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
    # print(train_queue)
    logging.info('prepare data finish! [%f]' % (time()-data_start))
    stored_arches = {} # log ging表示添加到记录中


    # 分不同的mode运行代码  default='random_single', help='search or single mode'
    print("args.mode: {}".format(args.mode)) # 应该只要运行一次即可
    
    # 这个模式下不进行训练，只对专门的数据集搜索所有可能的架构（135个）
    if args.mode == 'get_arch': #需要search的时候才使用
        # arch表示architecture，根据不同的数据集遍历不同的结构，可以保存在save下面
        args.arch_file = 'save/{}_arches.txt'.format(args.dataset)
        fw = open(args.arch_file, 'w') # 打开arch文件
        written_arches = []
        args.search_epochs = min(args.search_epochs, SPACE) 
        # default=1000, SPACE = 9 * 5 * 3 = 135
        while True:
            arch_single = sample_arch_cf() # sample简单的CF model表示的dict()
            print("arch_single: {}".format(arch_single))
            arch_encoding = '{}_{}_{}_{}_{}'.format(arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'])
            while arch_encoding in written_arches:
                arch_single = sample_arch_cf()
                arch_encoding = '{}_{}_{}_{}_{}'.format(arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'])
            fw.write(arch_encoding + '\n') # 写入文件
            written_arches.append(arch_encoding)
            if len(written_arches) == SPACE:
                print('finish...')
                fw.close()
                break
    
    # 大概是explicit的情况下才使用
    if args.mode == 'random_cf_signal': 
        search_start = time()
        performance = {}
        best_arch, best_rmse = None, 100000
        arch_batch_size = 8   
        arch_cf_list = []
        for search_epoch in range(args.search_epochs):
            if len(performance.keys()) == 130:
                break
            for arch_index in range(arch_batch_size):
                arch_cf = sample_arch_cf_signal()
                while str(arch_cf) in performance.keys():
                    arch_cf = sample_arch_cf_signal()
                arch_cf_list.append(arch_cf)
            arch_start = time()
            avaliable_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            assigned_device_ids = [k % len(avaliable_device_ids) for k in range(len(arch_cf_list))]
            hyper_parameters = [[0.1, 0]]
            with mp.Pool(processes=len(arch_cf_list)) as p:
                p.name = 'test'
                jobs = [[arch_cf_list[i], num_users, num_items, train_queue, valid_queue, test_queue, args, hyper_parameters[0], assigned_device_ids[i]] for i in range(len(arch_cf_list))]
                rmse_list = p.map(get_hyperparam_performance, jobs)
                p.close()
            for k in range(arch_batch_size):
                arch = str(arch_cf_list[k])
                performance_arch = rmse_list[k]
                logging.info('search_epoch: %d, arch: %s, rmse: %.4f, arch spent: %.4f' % (search_epoch, str(arch), performance_arch, time()-arch_start))

    # 单个模型的NAS，采用random search，第一阶段用不到
    if args.mode == 'random_single':
        
        search_start = time()
        performance = {}
        best_arch, best_rmse = None, 100000
        arch_batch_size = 1  
        args.search_epochs = min(args.search_epochs, SPACE)
    
        remaining_arches_encoding = open(args.remaining_arches, 'r').readlines()
        remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
        if not args.arch_assign:
            remaining_arches_encoding = remaining_arches_encoding
        else:
            start, end = eval(args.arch_assign)
            remaining_arches_encoding = remaining_arches_encoding[start:end]
        arch_count = 0
        while True:
            if arch_count >= len(remaining_arches_encoding):
                break
            # sample an arch
            arch_encoding = remaining_arches_encoding[arch_count]
            arch_single = sample_arch_cf()
            arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
            arch_count += 1
            performance[str(arch_single)] = 0
            if len(performance) >= len(remaining_arches_encoding):
                break


            arch_start = time()
            avaliable_device_ids = [0,1,2,3]
            hostname = socket.gethostname()
            print("hostname: {}".format(hostname))
            if hostname == 'rl3':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'fib-dl':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'fib-dl3':
                avaliable_device_ids = [2,3,5]
            elif hostname == 'fib':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname =='rl2':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'abc':
                avaliable_device_ids = [4,5,6,7]
            else:
                pass

            lr_candidates = [0.01, 0.02, 0.05, 0.1]
            rank_candidates = [2, 4, 8, 16]
            hyper_parameters = list(product(lr_candidates, rank_candidates))
            run_one_model = 0


            while True:
                avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
                if hostname == 'fib':
                    avaliable_device_ids = [0, 1, 2, 3]
                elif hostname == 'abc':
                    avaliable_device_ids = [4,5,6,7]
                else:
                    pass
                print(avaliable_device_ids)
                assigned_device_ids = avaliable_device_ids
                if run_one_model > 0:
                    break
                task_number = math.ceil(16 / len(avaliable_device_ids)) 
                task_split = list(range(0, 16, len(avaliable_device_ids)))
                task_split.append(16)
                task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]
                for tasks in task_index:
                    with mp.Pool(processes=len(tasks)) as p:
                        print('Stage1')
                        p.name = 'test'
                        if args.data_type == 'implicit':
                            jobs = [[arch_single, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, hyper_parameters[i], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                        else:
                            jobs = [[arch_single, num_users, num_items, train_queue, valid_queue, test_queue, args, hyper_parameters[i], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                        rmse_list1 = p.map(get_single_model_performance, jobs)
                        run_one_model += 1
                        p.close()

                for k in range(len(hyper_parameters)):
                    arch_encoding = '{}_{}_{}_{}_{}'.format(arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'])

    # 不搜索，采用单独的模型进行分析
    elif args.mode == 'DELF' or 'SVD_plus_plus' or 'FISM' or 'SVD' or 'GMF' or 'MLP' or 'CML' or 'JNCF_Dot' or 'JNCF_Cat' or 'DMF':
        # 根据参数选取对应的一个模型
        start = time()
        print('Debug in model {}'.format(args.mode))
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
        else:
            print('other mode...')
            pass
        if args.use_gpu:
            model = model.cuda()
        if args.mode == 'test_mlp':
            optimizer = torch.optim.Adagrad(model.train_parameters(), args.lr) # 默认Adagrad优化， 可以进行分析
        elif args.opt == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
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

        for train_epoch in range(args.train_epochs):
            # 训练
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
            performances.append(rmse)
            if args.data_type == 'explicit':
                logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (train_epoch, loss, rmse, time()-start))
            elif args.data_type == 'implicit':
                # logging.info('train_epoch: %d, loss: %.4f, recall20: %.4f, ndcg20: %.4f[%.4f]' % (train_epoch, loss, rmse,ndcg, time()-start))
                logging.info('train_epoch: %d, loss: %.4f, recall20: %.4f[%.4f]' % (train_epoch, loss, rmse, time()-start))
            else:
                pass
            # print(train_epoch)
            # print(loss)
            # print(rmse)
            
            writer.add_scalar('train/loss', loss, train_epoch) #tensorboard
            writer.add_scalar('train/rmse', rmse, train_epoch)
    else:
        print('No such mode.')
    print(save_name)
