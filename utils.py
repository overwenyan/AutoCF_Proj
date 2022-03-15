# import torch
from audioop import avg
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import groupby
from scipy.stats import gaussian_kde
import seaborn as sns
import random
import json
from itertools import combinations
from scipy import stats


def get_loss_recall(filename, train_epochs=2000, savedir='random_mode'):
    # filename = 'GMF_ml-1m_2_Adagrad0.05_1e-05_1_2022-01-29-00:29:08.txt'
    txtfile_dir = os.path.join(savedir, os.path.join('log',  filename))
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
    plt.figure(figsize=(12, 4), dpi=100)
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

def violin_plot(groups, xlabel=['Adagrad', 'Adam', 'SGD'], figtitle=''):
    # fig, axes = plt.subplots(figsize=(12, 5))
    fig, axes = plt.subplots()
    sample_size = groups.shape[1]

    groups_num = groups.shape[0]
    all_data = np.zeros((sample_size, groups_num), dtype=np.int32)
    for i in range(sample_size):
        all_data[i,:] = np.argsort(-groups[:,i]) + 1

    # print("data len: {}, shape: {}".format(len(all_data), all_data[0].shape))
    axes.violinplot(all_data,
                    showmeans=False,
                    showmedians=True
                    )
    axes.set_title(figtitle + 'violin plot')

    # # adding horizontal grid lines
    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data.T))], )
    axes.set_yticks(list(range(1,1+len(xlabel))), )

    axes.set_xlabel(figtitle)
    axes.set_ylabel('ranking distribution')

    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data.T))],
            xticklabels=xlabel
            # ,yticklabels=list(range(1,1+len(xlabel)))
            )

    plt.show()
    plt.close()
    return fig

def get_rank_from_score(groups):
    # fig, axes = plt.subplots(figsize=(8, 5))
    # fig, axes = plt.subplots()
    sample_size = groups.shape[1]
    groups_num = groups.shape[0]
    all_data = np.zeros((sample_size, groups_num), dtype=np.int32)
    for i in range(sample_size):
        all_data[i,:] = np.argsort(-groups[:,i]) + 1

    # print("data len: {}, shape: {}".format(len(all_data), all_data[0].shape))
    return all_data

def get_srcc_from_rank(groups, ranks, optslist=['Adagrad', 'Adam', 'SGD']):
    configsize = len(ranks)
    optset = list(range(ranks.shape[1]))
    srcc_list = []
    srcc2_list = []
    # for (opt1, opt2) in zip(optset, optset[1:] + optset[:1]):
    for (opt1, opt2) in list(combinations(optset,2)):
        # print(optslist[opt1], optslist[opt2])
        # print(ranks[0,opt1], ranks[0,opt2])
        sumranktmp = 0
        for cnt in range(configsize):
            sumranktmp += pow(ranks[cnt,opt1] - ranks[cnt,opt2], 2)
        # print("sumranktmp: {}, configsize: {}".format(sumranktmp, configsize))
        srcc = 1 - 6*sumranktmp/(configsize*(configsize**2-1))
        srcc_list.append(srcc)

        # print(ranks[:,opt1], ranks[:,opt2])
        srcc2 = stats.spearmanr(groups[opt1,:], groups[opt2,:])
        # srcc2_list.append(abs(srcc2.correlation))
        srcc2_list.append(srcc2.correlation)
        # print(optslist[opt1], optslist[opt2], srcc, srcc2[0])

    # avg_srcc = np.mean(srcc_list)
    avg_srcc = np.mean(srcc2_list)
    return avg_srcc

def hp_boxplot(groups, xlabel, figtitle):
    plt.figure()
    plt.boxplot(groups.T, labels=xlabel)
    # sns.boxplot(data=groups, labels=xlabel)
    # plt.boxplot((lr_result_list_dict['0.001'],lr_result_list_dict['0.005'], \
    #         lr_result_list_dict['0.01'],lr_result_list_dict['0.05'],\
    #         lr_result_list_dict['0.1'],lr_result_list_dict['0.5'],\
    #         lr_result_list_dict['1.0'],lr_result_list_dict['1.5'],
    #         lr_result_list_dict['1.5']),\
            # labels=('0.001','0.005', '0.01','0.05', '0.1', '0.5', '1.0','1.5','2.0'))
    
    plt.title(figtitle + 'boxplot')
    plt.xlabel(figtitle)
    plt.ylabel('recall@20')
    fig = plt.gcf()
    # plt.savefig(os.path.join('box_violin_plot', os.path.join(figtitle + 'boxplot')))
    plt.show()
    # plt.close()
    return fig

def get_srcc_from_rank2(ranks):
    configsize = len(ranks)
    optset = list(range(ranks.shape[1]))
    srcc_list = []
    # for (opt1, opt2) in zip(optset, optset[1:] + optset[:1]):
    for (opt1, opt2) in list(combinations(optset,2)):
        srcc =  stats.spearmanr(ranks[:,opt1], ranks[:,opt2])
        srcc_list.append((srcc))
    avg_srcc = np.mean(srcc_list)
    return avg_srcc