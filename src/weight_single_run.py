import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import datetime

if __name__ == '__main__':
    logfilePath = './weight_random/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    
    # for lr test random run
    anchor_config_num = 5
    lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    # weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=anchor_config_num)

    gpu_num = 7
    # dataset = 'ml-1m'
    dataset = 'ml-100k'
    weight_decay_result_list_dict = {'1e-06': [], '1e-05':[], '0.0001':[], '0.001':[], '0.01':[]}
    # best_scores = np.zeros((3, anchor_config_num))
    for i in range(anchor_config_num):
        for weight_decay in ['1e-06','1e-05', '0.0001','0.001','0.01']:
            os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} --lr {:.4f} --embedding_dim {} --weight_decay {:.6f}'.format(dataset, gpu_num, gpu_num, logfilePath, opt_list[i], lr_list[i], embedding_dim_list[i], float(weight_decay)))

            filenametmp = 'GMF'+'_'+ dataset + '_' + str(embedding_dim_list[i]) \
                    +  '_' + str(opt_list[i]) + ('%.4f'%(lr_list[i])) \
                    +  '_implicit_' + ('%.6f'%(float(weight_decay)))

            filePath_list = []
            for ii,jj,kk in os.walk(logfilePath + 'log'):
                filePath_list = kk       
            
            # get rank and save 
            for fp in filePath_list:
                cnt = 0
                if fp.startswith(filenametmp):
                    # print('debug1')
                    cnt += 1
                    print('weight decay find {}, filename = {}'.format(cnt, fp))
                    epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp, train_epochs=2000, savedir='weight_random')
                    recall20_array = np.array(recall20_list)
                    max_epoch  = np.argmax(recall20_array)
                    max_recall = np.max(recall20_array)
                    weight_decay_result_list_dict[weight_decay].append(max_recall)
                    print("fname: {}, max_recall: {}".format(fp, max_recall))
                    
    print('weight_decay_result_list_dict: {}'.format(weight_decay_result_list_dict))

    # dumps 将数据转换成字符串
    info_json = json.dumps(weight_decay_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    # f = open(logfilePath+'embinfo23.json', 'w')
    f = open(logfilePath+'weighdeinfo7.json', 'w')
    f.write(info_json)