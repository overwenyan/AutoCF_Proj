import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from models import Network_Single_CF_Signal, single_model
import logging
from time import time, sleep
import setproctitle
import sys
import os
import random
from tensorboardX import SummaryWriter
import math


def train_single_cf_efficiently(train_queue, model, optimizer, args, sparse=''):
    if args.loss_func == 'logloss' or args.data_type == 'explicit':
        torch.manual_seed(args.seed)
        users_train, items_train, labels_train, users_ratings_train, items_ratings_train = train_queue
        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        inferences, regs = model(users_train, items_train, users_ratings_train, items_ratings_train)
        loss = model.compute_loss(inferences, labels_train, regs)
        # loss = loss.to(torch.float32) # added
        loss.backward()
        optimizer.step()
        if args.use_gpu:
            return loss.cpu().detach().numpy().tolist()
        else:
            return loss.detach().numpy().tolist()
    elif args.loss_func == 'bprloss': # default
        torch.manual_seed(args.seed)
        users_train, items_train, negs_train, users_ratings_train, items_ratings_train = train_queue
        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        inferences, regs = model.forward_pair(users_train, items_train, negs_train, users_ratings_train, items_ratings_train)
        loss = model.compute_loss_pair(inferences, regs)
        loss.backward()
        optimizer.step()
        if args.use_gpu:
            return loss.cpu().detach().numpy().tolist()
        else:
            return loss.detach().numpy().tolist()
    else:
        raise NotImplementedError


def evaluate_cf_efficiently(model, test_queue, sparse=''):
    '''评估显式模型的方法，只运行一次，返回rmse'''
    model.eval()
    users, items, labels, users_ratings, items_ratings = test_queue
    inferences, _ = model(users, items, users_ratings, items_ratings)
    mse = F.mse_loss(inferences, torch.reshape(labels, [-1, 1]))
    rmse = torch.sqrt(mse)
    if 1 == 1:
        return rmse.cpu().detach().numpy().tolist()
    else:
        return rmse.detach().numpy().tolist()


def DCG(hit: torch.Tensor, topk: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
    return hit.sum(-1)

def IDCG(num_pos: int, topk: int) -> torch.Tensor:
    hit = torch.zeros(topk, dtype=torch.float)
    hit[:num_pos] = 1
    return DCG(hit, topk)

def get_rank_metrics(scores, ground_truth, topk, metric='Recall'):
    if metric == 'Recall':
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        _cnt = scores.shape[0] - (num_pos == 0).sum().item()
        _sum = (is_hit/(num_pos+1e-8)).sum().item()
        return _sum, _cnt
    else:
        IDCGs = torch.empty(1 + topk, dtype=torch.float)
        IDCGs[0] = 1  # avoid 0/0
        for i in range(1, topk + 1):
            IDCGs[i] = IDCG(i,topk)

        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1).clamp(0, topk).to(torch.long)
        print("is_hit: {}, size: {}".format(is_hit, is_hit.shape))
        print("topk: {}".format(topk))
        dcg = DCG(is_hit, topk, device)
        idcg = IDCGs[num_pos]
        ndcg = dcg/idcg.to(device)
        _cnt = scores.shape[0] - (num_pos == 0).sum().item()
        _sum = ndcg.sum().item()
        return sum / _cnt

def evaluate_cf_efficiently_implicit(model, test_queue, all_users, all_items, args):
    '''评估隐式模型的方法，只运行一次，返回recall@20'''
    model.eval()
    users, items, labels, users_ratings, items_ratings = test_queue
    users_ratings = users_ratings.cuda()
    items_ratings = items_ratings.cuda()
    inferences, _ = model(all_users, all_items, users_ratings, items_ratings)
    # inferences = inferences.cuda()
    inferences_reshaped = inferences.reshape(args.num_users, args.num_items)
    users_ratings = users_ratings.cuda()
    train_mask = users_ratings.to_sparse().to_dense() #待改正错误ok
    # print('train_mask: {}'.format(train_mask))
    final_inferences = inferences_reshaped - train_mask * 1e10
    users_test, items_test, labels_test = test_queue[0:3]
    # recall_20 = get_rank_metrics(final_inferences, labels, 20)
    recall_20_sum, recall_20_cnt = get_rank_metrics(final_inferences, labels, 20, metric='Recall')
    recall_20 = recall_20_sum / recall_20_cnt #modified
    # print("recall_20: {}".format(recall_20))

    # ndcg20 = get_rank_metrics(final_inferences, labels, 20, metric='NDCG')
    # print("ndcg20: {}".format(ndcg20))
    return recall_20
    # return recall_20, ndcg20
    
def evaluate_cf_efficiently_implicit_minibatch(model, test_queue, args, num_items, device):
    model.eval()
    users, items, labels, users_ratings, items_ratings = test_queue
    batch_size = 5000
    batch_num = math.ceil(len(users) / batch_size)
    recall_20_avg_cnt, recall_20_avg_sum, ndcg_20_avg_cnt, ndcg_20_avg_sum = 0, 0, 0, 0
    for k in range(batch_num):
        start = k * batch_size
        end = (k+1)*batch_size if (k+1)*batch_size <= len(users) else len(users)
        user_batch = users[start:end]
        items_batch = items[start:end]
        labels_batch = labels[start:end]
        if end >= len(users):
            end -= 1
        if users[start].cpu().detach().numpy().tolist() != users[end].cpu().detach().numpy().tolist():
            all_users = torch.tensor(list(range(users[start].cpu().detach().numpy().tolist(), users[end].cpu().detach().numpy().tolist())), dtype=torch.int64).repeat_interleave(num_items)
            all_items = torch.tensor(list(range(num_items)), dtype=torch.int64).repeat(users[end].cpu().detach().numpy().tolist()-users[start].cpu().detach().numpy().tolist())
            with torch.cuda.device(device):              
                all_users = all_users.cuda()
                all_items = all_items.cuda()
            inferences, _ = model(all_users, all_items, users_ratings, items_ratings)
            inferences_reshaped = inferences.reshape(users[end].cpu().detach().numpy().tolist()-users[start].cpu().detach().numpy().tolist(), args.num_items)
            train_mask = users_ratings.to_dense()[range(users[start].cpu().detach().numpy().tolist(), users[end].cpu().detach().numpy().tolist())]
            final_inferences = inferences_reshaped - train_mask * 1e10
            ground_truth_batch = labels[range(users[start].cpu().detach().numpy().tolist(), users[end].cpu().detach().numpy().tolist())]
            recall_20_sum, recall_20_cnt = get_rank_metrics(final_inferences, ground_truth_batch, 20)
            ndcg_20_sum, ndcg_20_cnt = get_rank_metrics(final_inferences, ground_truth_batch, 10)
        else:
            recall_20_sum, recall_20_cnt, ndcg_20_sum, ndcg_20_cn = 0, 0, 0, 0
        recall_20_avg_cnt += recall_20_cnt
        recall_20_avg_sum += recall_20_sum
        ndcg_20_avg_cnt += ndcg_20_cnt
        ndcg_20_avg_sum += ndcg_20_sum

    return recall_20_avg_sum / recall_20_avg_cnt, ndcg_20_avg_sum / ndcg_20_avg_cnt


def train_single_cf(train_queue, model, optimizer, args, sparse=''):
    torch.manual_seed(args.seed)
    users_train, items_train, labels_train, users_ratings_train, items_ratings_train, users_sparse_ratings_train, items_sparse_ratings_train = train_queue
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    if 'JNCF' in args.mode or sparse == 'ui':
        inferences, regs = model(
            users_train, items_train, users_sparse_ratings_train, items_sparse_ratings_train)
    elif sparse == 'u':
        inferences, regs = model(
            users_train, items_train, users_sparse_ratings_train, items_ratings_train)
    elif sparse == 'i':
        inferences, regs = model(
            users_train, items_train, users_ratings_train, items_sparse_ratings_train)
    else:
        inferences, regs = model(
            users_train, items_train, users_ratings_train, items_ratings_train)
    loss = model.compute_loss(inferences, labels_train, regs)
    loss.backward()
    optimizer.step()
    if args.use_gpu:
        return loss.cpu().detach().numpy().tolist()
    else:
        return loss.detach().numpy().tolist()


def train_single_cf_signal(train_queue, model, optimizer, args, sparse=''):
    users_train, items_train, labels_train, users_ratings_train, items_ratings_train, users_sparse_ratings_train, items_sparse_ratings_train = train_queue
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    inferences, regs = model(users_train, items_train, users_ratings_train,
                             items_ratings_train, users_sparse_ratings_train, items_sparse_ratings_train)
    loss = model.compute_loss(inferences, labels_train, regs)
    loss.backward()
    optimizer.step()
    if args.use_gpu:
        return loss.cpu().detach().numpy().tolist()
    else:
        return loss.detach().numpy().tolist()


def evaluate_cf(model, test_queue, sparse=''):
    model.eval()
    users, items, labels, users_ratings, items_ratings, users_sparse_ratings, items_sparse_ratings = test_queue
    if sparse == 'ui':
        inferences, _ = model(
            users, items, users_sparse_ratings, items_sparse_ratings)
    elif sparse == 'u':
        inferences, _ = model(
            users, items, users_sparse_ratings, items_ratings)
    elif sparse == 'i':
        inferences, _ = model(users, items, users_ratings,
                              items_sparse_ratings)
    else:
        inferences, _ = model(users, items, users_ratings, items_ratings)
    mse = F.mse_loss(inferences, torch.reshape(labels, [-1, 1]))
    rmse = torch.sqrt(mse)
    return rmse.cpu().detach().numpy().tolist()


def evaluate_cf_signal(model, test_queue, sparse=''):
    model.eval()
    users, items, labels, users_ratings, items_ratings, users_sparse_ratings, items_sparse_ratings = test_queue
    inferences, _ = model(users, items, users_ratings, items_ratings,
                          users_sparse_ratings, items_sparse_ratings)
    mse = F.mse_loss(inferences, torch.reshape(labels, [-1, 1]))
    rmse = torch.sqrt(mse)
    return rmse.cpu().detach().numpy().tolist()


def get_arch_performance_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device, if_valid):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w',
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    random_name = args.mode + '_' + args.dataset + '_sub_' + '%.2f' % random.random() + '_' + \
        str(param[0]) + '_' + str(param[1])
    arch_encoding = '{}_{}_{}_{}_{}'.format(
        arch['cf'], arch['emb']['u'], arch['emb']['i'], arch['ifc'], arch['pred'])
    fh = logging.FileHandler(os.path.join(
        'save/log_sub', args.mark + '_' + arch_encoding + '_' + random_name + '.txt'))
    logging.getLogger().addHandler(fh)
    logging.info(str(arch))
    torch.manual_seed(args.seed)
    # setproctitle.setproctitle('gaochen@get_performance{}'.format(int(device)))
    setproctitle.setproctitle('wenyan@get_performance{}'.format(int(device)))
    writer = SummaryWriter(
        log_dir='save/tensorboard_sub/{}_{}'.format(arch_encoding, random_name))
    lr = param[0]
    args.embedding_dim = param[1]
    print('lr', lr, 'rank', args.embedding_dim)
    if args.use_gpu:
        with torch.cuda.device(device):
            print('Using GPU {}'.format(device))
            model = single_model(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
            train_queue = [k.cuda() for k in train_queue]
            train_queue[3] = train_queue[3].to_sparse()
            train_queue[4] = train_queue[4].to_sparse()
            test_queue = [k.cuda() for k in test_queue]
            test_queue[3] = test_queue[3].to_sparse()
            test_queue[4] = test_queue[4].to_sparse()

    else:
        print('Does not use GPU')
        model = single_model(num_users, num_items, args.embedding_dim, arch, args.weight_decay)
        train_queue[3] = train_queue[3].to_sparse()
        train_queue[4] = train_queue[4].to_sparse()
        test_queue[3] = test_queue[3].to_sparse()
        test_queue[4] = test_queue[4].to_sparse()

    optimizer = torch.optim.Adagrad(model.parameters(), lr)
    losses = []
    performances = []
    start = time()
    for train_epoch in range(args.train_epochs):
        loss = train_single_cf_efficiently(train_queue, model, optimizer, args)
        losses.append(loss)
        if train_epoch > 1000:
            if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
                break
        performance = evaluate_cf_efficiently(model, test_queue)
        performances.append(performance)

        logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
            train_epoch, loss, performance, time()-start))
        writer.add_scalar('train/loss', loss, train_epoch)
        writer.add_scalar('train/rmse', performance, train_epoch)
    writer.close()
    return performance


def get_arch_performance_implicit_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, train_queue_pair, args, param, device, if_valid):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w',
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    random_name = args.mode + '_' + args.dataset + '_sub_' + '%.2f' % random.random() + '_' + \
        str(param[0]) + '_' + str(param[1])
    arch_encoding = '{}_{}_{}_{}_{}'.format(
        arch['cf'], arch['emb']['u'], arch['emb']['i'], arch['ifc'], arch['pred'])
    fh = logging.FileHandler(os.path.join(
        'save/log_sub', args.mark + '_' + arch_encoding + '_' + random_name + '.txt'))
    logging.getLogger().addHandler(fh)
    logging.info(str(arch))
    torch.manual_seed(args.seed)
    # setproctitle.setproctitle('gaochen@get_performance{}'.format(int(device)))
    setproctitle.setproctitle('wenyan@get_performance{}'.format(int(device)))
    writer = SummaryWriter(
        log_dir='save/tensorboard_sub/{}_{}'.format(arch_encoding, random_name))
    lr = param[0]
    args.embedding_dim = param[1]
    print('lr', lr, 'rank', args.embedding_dim)
    if args.use_gpu:
        with torch.cuda.device(device):
            print('Using GPU {}'.format(device))
            model = single_model(num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
            if args.data_type == 'explicit':
                train_queue = [k.cuda() for k in train_queue]
                train_queue[3] = train_queue[3].to_sparse()
                train_queue[4] = train_queue[4].to_sparse()
            if args.data_type == 'implicit':
                train_queue_pair = [k.cuda() for k in train_queue_pair]
                train_queue_pair[3] = train_queue_pair[3].to_sparse()
                train_queue_pair[4] = train_queue_pair[4].to_sparse()

    else:
        model = single_model(num_users, num_items, args.embedding_dim, arch, args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    losses = []
    performances = []
    start = time()
    all_users = torch.tensor(list(range(num_users)), dtype=torch.int64).repeat_interleave(num_items)
    all_items = torch.tensor(list(range(num_items)), dtype=torch.int64).repeat(num_users)
    with torch.cuda.device(device):
        all_users = all_users.cuda()
        all_items = all_items.cuda()
    with torch.cuda.device(device):
        test_queue = [k.cuda() for k in test_queue]
        test_queue[3] = test_queue[3].to_sparse()
        test_queue[4] = test_queue[4].to_sparse()

    for train_epoch in range(args.train_epochs):
        if args.data_type == 'implicit':
            loss = train_single_cf_efficiently(train_queue_pair, model, optimizer, args)
        else:
            loss = train_single_cf(train_queue, model, optimizer, args)
        losses.append(loss)
        stop_train = False
        if train_epoch > 1000:
            if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue_pair[0].shape[0] or np.isnan(losses[-1]) or train_epoch == 2000-1:
                stop_train = True
            if stop_train:
                performance = evaluate_cf_efficiently_implicit_minibatch(model, test_queue, args, num_items, device)
            else:
                performance = 0, 0

            logging.info('train_epoch: %d, loss: %.4f, recall20: %.4f recall10: %.4f [%.4f]' % (
                train_epoch, loss, performance[0], performance[1], time()-start))
            writer.add_scalar('train/loss', loss, train_epoch)
            writer.add_scalar('train/recall20', performance[0], train_epoch)
            writer.add_scalar('train/recall10', performance[1], train_epoch)
            if stop_train:
                break
    writer.close()
    return 0



def get_arch_performance_cf_signal_param_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device, if_valid=False):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w',
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(
        args.save, args.mode + '_subprocess_' + str(random.random()) + str(param) + '.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(str(arch))
    torch.manual_seed(args.seed)
    # setproctitle.setproctitle('gaochen@get_performance{}'.format(int(device)))
    setproctitle.setproctitle('wenyan@get_performance{}'.format(int(device)))
    lr = param[0]
    args.weight_decay = param[1]
    try:
        if args.use_gpu:
            with torch.cuda.device(device):
                print('Using GPU')
                model = Network_Single_CF_Signal(
                    num_users, num_items, args.embedding_dim, arch, args.weight_decay).cuda()
                train_queue = [k.cuda() for k in train_queue]
                test_queue = [k.cuda() for k in test_queue]
                if if_valid:
                    valid_queue = [k.cuda() for k in valid_queue]

        else:
            model = Network_Single_CF_Signal(
                num_users, num_items, args.embedding_dim, arch, args.weight_decay)

        optimizer = torch.optim.Adagrad(model.parameters(), lr)
        losses = []
        performances = []
        valid_performances = []
        start = time()
        for train_epoch in range(args.train_epochs):
            loss = train_single_cf_signal(train_queue, model, optimizer, args)
            losses.append(loss)
            if train_epoch > 100:
                if (losses[-2]-losses[-1])/losses[-1] < 1e-4/train_queue[0].shape[0] or np.isnan(losses[-1]):
                    break
                if if_valid:
                    if (valid_performances[-1] > valid_performances[-2] and valid_performances[-2] > valid_performances[-3]) or \
                        (valid_performances[-1] > valid_performances[-3] and valid_performances[-3] > valid_performances[-5]):
                        break
                    if sum([round(k, 4) for k in valid_performances[-10:]]) == 10 * round(valid_performances[-1], 4):
                        break
                else:
                    if (performances[-1] > performances[-2] and performances[-2] > performances[-3]) or \
                            (performances[-1] > performances[-3] and performances[-3] > performances[-5]):
                        break
                    if sum([round(k, 4) for k in performances[-10:]]) == 10 * round(performances[-1], 4):
                        break
            
            performance = evaluate_cf_signal(model, test_queue)
            performances.append(performance)
            if if_valid:
                valid_performance = evaluate_cf_signal(model, valid_queue)
                valid_performances.append(valid_performance)

            if args.mode == 'test':
                logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
                    train_epoch, loss, performance, time()-start))

            logging.info('train_epoch: %d, loss: %.4f, rmse: %.4f[%.4f]' % (
                train_epoch, loss, performance, time()-start))
    except Exception as e:
        print(e)
        performance = 1.0
    return performance
