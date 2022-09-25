# import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from itertools import groupby
import json
from utils import *

if __name__ == '__main__':
    logfilePath = './save/log'
    # plt.use('TkAgg')
    filePath_list = []
    for i,j,k in os.walk(logfilePath):
        filePath_list = k

    gmf_cnt = 0
    fig_res_dict = {'opt':{}, 
                    'lr':{}, 
                    'embedding_dim':{}, 
                    'weight_decay':{},
                    'data_type':{}}
    optresult_list_dict = {'Adagrad': [], 'Adam': [], 'SGD': []}
    lr_result_list_dict = {'0.001': [], '0.005':[], '0.01':[], '0.05':[], \
                        '0.1':[], '0.5':[],'1.0':[],'1.5':[],'2.0':[],}
    embedding_dim_result_list_dict = {'1': [], '2':[], '4':[], '8':[], '16':[], '32':[], '64':[]}
    weight_decay_result_list_dict = {'1e-06': [], '1e-05':[], '0.0001':[], '0.001':[], '0.01':[]}
    
    # implicit
    for fp in filePath_list:
        # tmpfilename='GMF_ml-1m_2_Adagrad0.05_1e-05_1_2022-01-29-00:29:08.txt'
        fp_item_dict = get_item_from_filename(fp)
        # print(fp_item_dict)
        # result_list = []
        # if  fp_item_dict['mode'] in ['GMF', 'FISM', 'MLP']  :
        if  (fp_item_dict['mode'] in ['GMF'] ) and (fp_item_dict['data_type'] in ['implicit']) and (fp_item_dict['dataset'] == 'ml-100k'):
            gmf_cnt += 1
            print("fp: {}".format(fp))
            epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp, train_epochs=2000, savedir='save')
            recall20_array = np.array(recall20_list)
            max_epoch  = np.argmax(recall20_array)
            max_recall = np.max(recall20_array)
            optresult_list_dict[fp_item_dict['opt']].append(max_recall)
            lr_result_list_dict[str(fp_item_dict['lr'])].append(max_recall)
            embedding_dim_result_list_dict[str(fp_item_dict['embedding_dim'])].append(max_recall)
            # weight_decay_result_list_dict[str(fp_item_dict['weight_decay'])].append(max_recall)

            print(fp_item_dict['opt'], fp_item_dict['lr'], max_epoch, max_recall)

            # print(fp_item_dict, max_epoch, max_recall)
            subtitle = json.dumps(fp_item_dict)
            fig = plot_loss_recall(epoch_list,bprloss_list,recall20_list,subtitle)
            figname = fp[:-4] + '.png'
            plt.savefig(os.path.join('save', os.path.join('datafig', figname)))
            plt.close()

    # opt analysis
    plt.figure()
    plt.boxplot((optresult_list_dict['Adagrad'],optresult_list_dict['Adam'], \
            optresult_list_dict['SGD']),labels=('Adagrad','Adam', 'SGD'))
    boxplotname = 'opt' + '_understanding' + 'implicit' + '5'+'.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

    # lr analysis
    plt.figure()
    plt.boxplot((lr_result_list_dict['0.001'],lr_result_list_dict['0.005'], \
            lr_result_list_dict['0.01'],lr_result_list_dict['0.05'],\
            lr_result_list_dict['0.1'],lr_result_list_dict['0.5'],\
            lr_result_list_dict['1.0'],lr_result_list_dict['1.5'],
            lr_result_list_dict['1.5']),\
            labels=('0.001','0.005', '0.01','0.05', '0.1', '0.5', '1.0','1.5','2.0'))
    boxplotname = 'lr' + '_understanding'  + 'implicit' + '5'+'.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

    # embedding_dim analysis
    plt.figure()
    plt.boxplot((embedding_dim_result_list_dict['1'],embedding_dim_result_list_dict['2'], \
            embedding_dim_result_list_dict['4'],embedding_dim_result_list_dict['8'],
            embedding_dim_result_list_dict['16'],embedding_dim_result_list_dict['32'],
            embedding_dim_result_list_dict['64']),\
                labels=('1','2', '4','8','16','32','64'))
    boxplotname = 'embedding_dim'+'_understanding' + 'implicit' + '5'+'.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

    '''
    # weight_decay analysis
    plt.figure()
    # weight_decay_result_list_dict = {'1e-6': [], '1e-5':[], '1e-4':[], '1e-3':[], '1e-2':[]}
    plt.boxplot((weight_decay_result_list_dict['1e-06'],weight_decay_result_list_dict['1e-05'], \
            weight_decay_result_list_dict['0.0001'],weight_decay_result_list_dict['0.001'],
            weight_decay_result_list_dict['0.01']),\
                labels=('1e-06','1e-05', '0.0001','0.001','0.01'))
    # weight_decay_result_list_dict = {'1e-6': [], '1e-5':[], '0.0001':[], '0.001':[], '0.01':[]}
    boxplotname = 'weight_decay' + '_understanding.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()
    boxplotname = 'mode_test.jpg'
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.show()
    '''
    


        