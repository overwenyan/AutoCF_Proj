# import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from itertools import groupby
import json



def get_loss_recall(filename, train_epochs=2000):
    # filename = 'GMF_ml-1m_2_Adagrad0.05_1e-05_1_2022-01-29-00:29:08.txt'
    txtfile_dir = os.path.join('save', os.path.join('log',  filename))
    file = open(txtfile_dir,encoding = "utf-8")
    # train_epochs = len(file.readlines())
    # train_epochs = 2000
    epoch_list = []
    bprloss_list = []
    recall20_list = []
    for i in range(train_epochs+2):
        # print(file.readline())
        lineinfo = file.readline()
        train_infolist = lineinfo.split(r', ').copy()
        # print(train_infolist)
        if i > 1:
            time_epoch = train_infolist[0].split(' ')
            # print(time_epoch)
            train_epoch = int(time_epoch[3])
            # print(train_epoch)
            loss = train_infolist[1].split(' ')
            # print(loss)
            bprloss = float(loss[1])
            # print(bprloss)
            recall20 = train_infolist[2].split(' ')[1].split('[')[0]
            recall20 = float(recall20)
            # print(train_epoch, epoch_list, recall20)
            epoch_list.append(train_epoch)
            bprloss_list.append(bprloss)
            recall20_list.append(recall20)
        else:
            pass
    return epoch_list,bprloss_list,recall20_list

def plot_loss_recall(epoch_list,bprloss_list,recall20_list, subtitle):
    plt.figure(figsize=(20, 4), dpi=100)
    # plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epoch_list, bprloss_list,color='deepskyblue')
    plt.xlabel('epoch')
    plt.ylabel('bprloss')
    # plt.show()

    # plt.figure(1)
    plt.subplot(1,2,2)
    plt.plot(epoch_list, recall20_list)
    plt.xlabel('epoch')
    plt.ylabel('recall@20')

    plt.suptitle(subtitle)
    fig = plt.show()
    return fig
    

def get_item_from_filename(fp):
    namelist = fp.split('_')
    # print(namelist)
    if namelist[0] == 'random' and namelist[1] == 'single':
        namelist[0] = 'random_single'
        del namelist[1:2]
    elif namelist[0] == 'JNCF' and namelist[1] == 'Dot':
        namelist[0] = 'JNCF_Dot'
        del namelist[1:2]
    mode = namelist[0]
    # print(namelist)
    dataset = namelist[1]
    embedding_dim = int(namelist[2])
    optim_lr = namelist[3]
    # print(optim_lr)
    optim_lr_list = [''.join(list(g)) for k, g in groupby(optim_lr, key=lambda x: x.isdigit())]
    optim = optim_lr_list[0]
    lr = ''.join(optim_lr_list[1:])#这里lr是string
    # print(lr)
    # lr = '{:e}'.format(float(lr))#科学计数法的string
    lr = float(lr)
    # print(lr)
    data_type = namelist[4]
    # print(data_type)
    weight_decay = float(namelist[5])
    seed = int(namelist[6])
    dateandtime = namelist[7].split('.')[0]
    ans = dict()
    ans['mode'] = mode
    ans['dataset'] = dataset
    ans['embedding_dim'] = embedding_dim
    ans['opt'] = optim
    ans['lr'] = lr
    ans['data_type'] = data_type
    ans['weight_decay'] = weight_decay
    ans['seed'] = seed
    ans['time'] = dateandtime
    # ans[]
    return ans

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
                    'weight_decay':{}}
    optresult_list_dict = {'Adagrad': [], 'Adam': [], 'SGD': []}
    lr_result_list_dict = {'0.001': [], '0.005':[], '0.01':[], '0.05':[]}
    embedding_dim_result_list_dict = {'1': [], '2':[], '4':[], '8':[]}
    # weight_decay_result_list_dict = {'1e-6': [], '1e-5':[], '1e-4':[], '1e-3':[], '1e-2':[]}
    weight_decay_result_list_dict = {'1e-06': [], '1e-05':[], '0.0001':[], '0.001':[], '0.01':[]}

    for fp in filePath_list:
        # tmpfilename='GMF_ml-1m_2_Adagrad0.05_1e-05_1_2022-01-29-00:29:08.txt'
        fp_item_dict = get_item_from_filename(fp)
        # print(fp_item_dict)
        # result_list = []
        if  fp_item_dict['mode'] == 'GMF'  :
            gmf_cnt += 1
            epoch_list,bprloss_list,recall20_list = get_loss_recall(filename=fp)
            recall20_array = np.array(recall20_list)
            max_epoch  = np.argmax(recall20_array)
            max_recall = np.max(recall20_array)
            optresult_list_dict[fp_item_dict['opt']].append(max_recall)
            lr_result_list_dict[str(fp_item_dict['lr'])].append(max_recall)
            embedding_dim_result_list_dict[str(fp_item_dict['embedding_dim'])].append(max_recall)
            weight_decay_result_list_dict[str(fp_item_dict['weight_decay'])].append(max_recall)

            print(fp_item_dict['opt'], fp_item_dict['lr'], max_epoch, max_recall)

            # print(fp_item_dict, max_epoch, max_recall)
            subtitle = json.dumps(fp_item_dict)
            fig = plot_loss_recall(epoch_list,bprloss_list,recall20_list,subtitle)
            figname = fp[:-4] + '.png'
            plt.savefig(os.path.join('save', os.path.join('datafig', figname)))
            plt.close()

    plt.figure()
    plt.boxplot((optresult_list_dict['Adagrad'],optresult_list_dict['Adam'], \
            optresult_list_dict['SGD']),labels=('Adagrad','Adam', 'SGD'))
    boxplotname = 'opt' + '_understanding.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

    plt.figure()
    plt.boxplot((lr_result_list_dict['0.001'],lr_result_list_dict['0.005'], \
            lr_result_list_dict['0.01'],lr_result_list_dict['0.05']),\
                labels=('0.001','0.005', '0.01','0.05'))
    boxplotname = 'lr' + '_understanding.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

    plt.figure()
    plt.boxplot((embedding_dim_result_list_dict['1'],embedding_dim_result_list_dict['2'], \
            embedding_dim_result_list_dict['4'],embedding_dim_result_list_dict['8']),\
                labels=('1','2', '4','8'))
    boxplotname = 'embedding_dim' + '_understanding.jpg'
    plt.title(boxplotname)
    plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.close()

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

    # boxplotname = 'mode_test.jpg'
    # plt.savefig(os.path.join('save', os.path.join('boxplot', boxplotname)))
    plt.show()


        

        



    