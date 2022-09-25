import numpy as np
import matplotlib.pyplot as plt
import os
# from itertools import groupby
# from scipy.stats import gaussian_kde
# import seaborn as sns
# import random
from utils import *
import datetime

if __name__ == '__main__':
    logfilePath = './opt_random/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # for opt test random run
    anchor_config_num = 15
    lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-6, high=1e-2, size=anchor_config_num)
    
    gpu_num = 6
    dataset = 'ml-1m'
    # dataset = 'ml-100k'
    opt_result_list_dict = {'Adagrad': [], 'Adam': [], 'SGD': []}
    # best_scores = np.zeros((3, anchor_config_num))
    for i in range(anchor_config_num):
        for opt in ['Adagrad', 'Adam', 'SGD']:
            os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} --lr {:.4f} --embedding_dim {} --weight_decay {:.6f}'.format(dataset, gpu_num, gpu_num, logfilePath, opt, lr_list[i], int(embedding_dim_list[i]), weight_decay_list[i]))

            filenametmp = 'GMF'+'_'+ dataset + '_' + str(embedding_dim_list[i]) \
                    +  '_' + opt + ('%.4f'%(lr_list[i])) \
                    +  '_implicit_' + ('%.6f'%(weight_decay_list[i]))

            filePath_list = []
            for ii,jj,kk in os.walk(logfilePath + 'log'):
                filePath_list = kk       
            
            # get rank and save 
            for fp in filePath_list:
                if fp.startswith(filenametmp):
                    # print('debug1')
                    print('f: {}'.format(fp))
                    epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp, train_epochs=2000, savedir=logfilePath)
                    recall20_array = np.array(recall20_list)
                    max_epoch  = np.argmax(recall20_array)
                    max_recall = np.max(recall20_array)
                    opt_result_list_dict[opt].append(max_recall)
                    print("fp: {}, max_recall: {}".format(fp, max_recall))
                    
    print('opt_result_list_dict: {}'.format(opt_result_list_dict))

    # dumps 将数据转换成字符串
    info_json = json.dumps(opt_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    f = open(logfilePath + 'optinfo35.json', 'w')
    f.write(info_json)
    # echo anchor_config_num