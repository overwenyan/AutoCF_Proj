import numpy as np
import matplotlib.pyplot as plt
import os
# from itertools import groupby
# from scipy.stats import gaussian_kde
# import seaborn as sns
# import random
from utils import *
import datetime
import _thread
import time
import argparse
import threading

parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
# parser.add_argument('--arch_lr', type=float, default=0.05, help='learning rate for arch encoding')
# parser.add_argument('--controller_lr', type=float, default=1e-1, help='learning rate for controller')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
# parser.add_argument('--update_freq', type=int, default=1, help='frequency of updating architeture')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--use_gpu', type=int, default=1, help='whether use gpu')
# parser.add_argument('--minibatch', type=int, default=1, help='whether use minibatch')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--train_epochs', type=int, default=2000, help='num of training epochs')
# parser.add_argument('--search_epochs', type=int, default=1000, help='num of searching epochs')
parser.add_argument('--save', type=str, default='save/', help='experiment name')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='random_single', help='search or single mode')
# parser.add_argument('--process_name', type=str, default='AutoCF@wenyan', help='process name')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
# parser.add_argument('--controller', type=str, default='PURE', help='structure of controller')
# parser.add_argument('--controller_batch_size', type=int, default=4, help='batch size for updating controller')
# parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
# parser.add_argument('--max_batch', type=int, default=65536, help='max batch during training')
# parser.add_argument('--device', type=int, default=0, help='GPU device')
# parser.add_argument('--multi', type=int, default=0, help='using multi-training for single architecture')
# parser.add_argument('--if_valid', type=int, default=1, help='use validation set for tuning single architecture or not')
# parser.add_argument('--breakpoint', type=str, default='save/log.txt', help='the log file storing existing results')
# parser.add_argument('--arch_file', type=str, default='src/arch.txt', help='all arches')
# parser.add_argument('--remaining_arches', type=str, default='src/arch.txt', help='')
# parser.add_argument('--arch_assign', type=str, default='[0,3]', help='')
parser.add_argument('--data_type', type=str, default='implicit', help='explicit or implicit(default)')
# parser.add_argument('--loss_func', type=str, default='bprloss', help='Implicit loss function')
# parser.add_argument('--mark', type=str, default='') # 

args = parser.parse_args()
opt_result_list_dict = {'Adagrad': [], 'Adam': [], 'SGD': []}

threadLock = threading.Lock()
threads = []
class myThread (threading.Thread):
    def __init__(self, threadID, name, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.counter = counter
        self.args = args
    def run(self):
        print ("Starting " + self.name)
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        threadLock.acquire()
        opt_run(self.name, self.args)
        # 释放锁
        threadLock.release()

def opt_run(threadName, args):
    count = 0
    delay = 0.002
    print( "%s: %s" % ( threadName, time.ctime(time.time()) ) )
    # while count < 5:
    #     # time.sleep(delay)
    #     count += 1
    #     print( "%s: %s" % ( threadName, time.ctime(time.time()) ) )
    
    '''
    os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} --lr {:.4f} --embedding_dim {} --weight_decay {:.6f}'.format(args.dataset, args.gpu, args.gpu, args.save, args.opt, args.lr, args.embedding_dim,  args.weight_decay))

    filenametmp = 'GMF'+'_'+ args.dataset + '_' + str(args.embedding_dim) \
                    +  '_' + args.opt + ('%.4f'%(args.lr)) \
                    +  '_implicit_' + ('%.6f'%(args.weight_decay))

    filePath_list = []
    for ii,jj,kk in os.walk(args.save + 'log'):
        filePath_list = kk       
    
    # get rank and save 
    for fp in filePath_list:
        if fp.startswith(filenametmp):
            # print('debug1')
            print('f: {}'.format(fp))
            epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp, train_epochs=2000, savedir=args.save)
            recall20_array = np.array(recall20_list)
            # max_epoch  = np.argmax(recall20_array)
            max_recall = np.max(recall20_array)
            opt_result_list_dict[args.opt].append(max_recall)
            print("fp: {}, max_recall: {}".format(fp, max_recall))
    '''
        

if __name__ == '__main__':
    logfilePath = './opt_random/'  
    args.save = logfilePath
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    
    # for opt test random run
    anchor_config_num = 2
    lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-6, high=1e-2, size=anchor_config_num)
    
    gpu_num = 7
    # dataset = 'ml-1m'
    args.dataset = 'ml-100k'
    # best_scores = np.zeros((3, anchor_config_num))
    for i in range(anchor_config_num):
        args.lr = lr_list[i]
        args.embedding_dim = embedding_dim_list[i]
        args.weight_decay = weight_decay_list[i]
        for opt in ['Adagrad', 'Adam', 'SGD']:
            args.gpu = (gpu_num + 1) % 8
            args.opt = opt
            _thread.start_new_thread( opt_run, ("Thread-" + opt, args,) )

            
                    
    print('opt_result_list_dict: {}'.format(opt_result_list_dict))

    # dumps 将数据转换成字符串
    info_json = json.dumps(opt_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    f = open(logfilePath + 'optinfo11.json', 'w')
    f.write(info_json)
    # echo anchor_config_num