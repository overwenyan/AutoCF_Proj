import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import datetime

if __name__ == '__main__':
    logfilePath = './lr_random/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    
    # for lr test random run
    anchor_config_num = 10
    # lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=anchor_config_num)

    gpu_num = 5
    dataset = 'ml-100k'
    # dataset = 'ml-1m'
    # optresult_list_dict = {'Adagrad': [], 'Adam': [], 'SGD': []}
    # lr_hp_list = ['0.001','0.005', '0.01','0.05', '0.1', '0.5', '1.0','1.5','2.0', '10.0', '20.0','50.0', '100.0', '1000.0']
    # lr_result_list_dict = {'0.001': [], '0.005':[], '0.01':[], '0.05':[], \
    #                     '0.1':[], '0.5':[],'1.0':[],'1.5':[],'2.0':[],'10.0':[], \
    #                         '20.0': [], '50.0': [], '100.0':[], '1000.0':[]}
    lr_hp_list = ['1e-6','1e-5', '1e-4','1e-3', '1e-2', '1e-1', '1e0','1e1','1e2', '1e3', '1e4', '1e5']
    lr_hp_list = [str(float(lr)) for lr in lr_hp_list]
    lr_result_list_dict = {}
    for lr in lr_hp_list:
        lr_result_list_dict[lr] = []
    
    
    best_scores = np.zeros((3, anchor_config_num))
    for i in range(anchor_config_num):
        for lr in lr_hp_list:
            os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} --lr {} --embedding_dim {} --weight_decay {:.4f}'.format(dataset, gpu_num, gpu_num, logfilePath, opt_list[i], float(lr), embedding_dim_list[i], weight_decay_list[i]))

            filenametmp = 'GMF'+'_'+ dataset + '_' + str(embedding_dim_list[i]) \
                    +  '_'+str(opt_list[i]) + str(float(lr)) \
                    +  '_implicit_' + ('%.4f'%(weight_decay_list[i]))

            filePath_list = []
            for ii,jj,kk in os.walk(logfilePath + 'log'):
                filePath_list = kk       
            
            # get rank and save 
            for fp in filePath_list:
                cnt = 0
                if fp.startswith(filenametmp):
                    # print('debug1')
                    cnt += 1
                    print('lr find {}, filename = {}'.format(cnt, fp))
                    epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp, train_epochs=2000, savedir='lr_random')
                    recall20_array = np.array(recall20_list)
                    max_epoch  = np.argmax(recall20_array)
                    max_recall = np.max(recall20_array)
                    lr_result_list_dict[lr].append(max_recall)
                    print("fn: {}, max_recall: {}".format(fp, max_recall))
                    
    print('lr_result_list_dict: {}'.format(lr_result_list_dict))

    # dumps 将数据转换成字符串
    info_json = json.dumps(lr_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    f = open(logfilePath+'lrinfo'+'102'+'.json', 'w')
    # f = open(logfilePath+'lrinfo21.json', 'w')
    f.write(info_json)