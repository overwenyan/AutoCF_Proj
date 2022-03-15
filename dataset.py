import torch
import torch.utils.data
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp
import time


class Data(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_queue(users, items, labels, batch_size):
    data = list(zip(users, items, labels))
    return torch.utils.data.DataLoader(Data(data), batch_size=batch_size, num_workers=8)


def get_data_queue(data_path, args):
    '''组装train_queue, valid_queue, test_queue'''
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'
    data_path = 'data/' + data_path

    if 'youtube' not in args.dataset:
        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if args.dataset == 'ml-100k':
                    line = line.split()
                elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                    line = line.split('::')
                elif args.dataset == 'ml-20m':
                    if i == 0:
                        continue
                    line = line.split(',')

                users.append(int(line[0]) - 1)
                items.append(int(line[1]) - 1)
                labels.append(float(line[2]))

        users, items, labels = shuffle(users, items, labels)
        num_train = int(len(users) * args.train_portion)
        num_valid = int(len(users) * args.valid_portion)

        if not args.mode == 'libfm':
            if not args.minibatch:
                train_queue = [torch.tensor(users[:num_train]).cuda(),
                               torch.tensor(items[:num_train]).cuda(),
                               torch.tensor(labels[:num_train]).cuda().float()]
            else:
                train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    torch.tensor(users[:num_train]), torch.tensor(
                        items[:num_train]),
                    torch.tensor(labels[:num_train])), batch_size=4096)

            valid_queue = [torch.tensor(users[num_train:num_train+num_valid]).cuda(),
                           torch.tensor(
                               items[num_train:num_train+num_valid]).cuda(),
                           torch.tensor(labels[num_train:num_train+num_valid]).cuda().float()]
            test_queue = [torch.tensor(users[num_train+num_valid:]).cuda(),
                          torch.tensor(items[num_train+num_valid:]).cuda(),
                          torch.tensor(labels[num_train+num_valid:]).cuda().float()]

        else:
            train_queue, valid_queue, test_queue = [], [], []
            for i in range(len(users)):
                if i < num_train:
                    train_queue.append(
                        {'user': str(users[i]), 'item': str(items[i])})
                elif i >= num_train and i < num_train+num_valid:
                    valid_queue.append(
                        {'user': str(users[i]), 'item': str(items[i])})
                else:
                    test_queue.append(
                        {'user': str(users[i]), 'item': str(items[i])})

            v = DictVectorizer()
            train_queue = [v.fit_transform(
                train_queue), np.array(labels[:num_train])]
            test_queue = [v.transform(test_queue), np.array(
                labels[num_train+num_valid:])]

    else: # for you tube
        [ps, qs, rs, labels] = np.load(data_path).tolist()
        labels = StandardScaler().fit_transform(
            np.reshape(labels, [-1, 1])).flatten().tolist()
        ps, qs, rs, labels = shuffle(ps, qs, rs, labels)
        num_train = int(len(ps) * args.train_portion)
        num_valid = int(len(ps) * args.valid_portion)

        print(len(set(ps)), max(ps), min(ps))
        print(len(set(qs)), max(qs), min(qs))
        print(len(set(rs)), max(rs), min(rs))
        print(len(ps))

        if not args.mode == 'libfm':
            if not args.minibatch:
                train_queue = [torch.tensor(ps[:num_train]).cuda().long(),
                               torch.tensor(qs[:num_train]).cuda().long(),
                               torch.tensor(rs[:num_train]).cuda().long(),
                               torch.tensor(labels[:num_train]).cuda().float()]
            else:
                train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    torch.tensor(ps[:num_train]).long(), torch.tensor(
                        qs[:num_train]).long(),
                    torch.tensor(rs[:num_train]).long(), torch.tensor(labels[:num_train])), batch_size=4096)

            valid_queue = [torch.tensor(ps[num_train:num_train+num_valid]).cuda().long(),
                           torch.tensor(
                               qs[num_train:num_train+num_valid]).cuda().long(),
                           torch.tensor(
                               rs[num_train:num_train+num_valid]).cuda().long(),
                           torch.tensor(labels[num_train:num_train+num_valid]).cuda().float()]
            test_queue = [torch.tensor(ps[num_train+num_valid:]).cuda().long(),
                          torch.tensor(qs[num_train+num_valid:]).cuda().long(),
                          torch.tensor(rs[num_train+num_valid:]).cuda().long(),
                          torch.tensor(labels[num_train+num_valid:]).cuda().float()]

        else:
            train_queue, valid_queue, test_queue = [], [], []
            for i in range(len(ps)):
                if i < num_train:
                    train_queue.append(
                        {'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})
                elif i >= num_train and i < num_train+num_valid:
                    valid_queue.append(
                        {'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})
                else:
                    test_queue.append(
                        {'p': str(ps[i]), 'q': str(qs[i]), 'r': str(rs[i])})

            v = DictVectorizer()
            train_queue = [v.fit_transform(
                train_queue), np.array(labels[:num_train])]
            test_queue = [v.transform(test_queue), np.array(
                labels[num_train+num_valid:])]

    return train_queue, valid_queue, test_queue


def get_data_queue_cf(data_path, args):
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'
    elif args.dataset == 'lastfm':
        data_path += 'user_artists.dat'

    data_path = 'data/' + data_path

    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if args.dataset == 'ml-100k':
                line = line.split()
            elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                line = line.split('::')
            elif args.dataset == 'ml-20m':
                if i == 0:
                    continue
                line = line.split(',')
            elif args.dataset == 'lastfm':
                if i == 0:
                    continue
                line = line.split()
            user = int(line[0]) - 1
            item = int(line[1]) - 1
            label = float(line[2])
            users.append(user)
            items.append(item)
            labels.append(label)

    labels = StandardScaler().fit_transform(
        np.reshape(labels, [-1, 1])).flatten().tolist()

    users, items, labels = shuffle(users, items, labels)
    num_train = int(len(users) * args.train_portion)
    num_valid = int(len(users) * args.valid_portion)

    num_users = max(users) + 1
    num_items = max(items) + 1

    if not args.minibatch:
        user_ratings, item_ratings, user_sparse_ratings, item_sparse_ratings = \
            convert_records_to_rating(
                users[:num_train], items[:num_train], labels[:num_train], num_users, num_items)
        train_queue = [torch.tensor(users[:num_train]),
                       torch.tensor(items[:num_train]),
                       torch.tensor(labels[:num_train]),
                       user_ratings, item_ratings, user_sparse_ratings, item_sparse_ratings
                       ]
    else:
        train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(users[:num_train]), torch.tensor(items[:num_train]),
            torch.tensor(labels[:num_train])), batch_size=4096)
    valid_user_ratings, valid_item_ratings, valid_user_sparse_ratings, valid_item_sparse_ratings = \
        convert_records_to_rating(users[num_train:num_train+num_valid],
                                  items[num_train:num_train+num_valid],
                                  labels[num_train:num_train+num_valid], num_users, num_items)

    valid_queue = [torch.tensor(users[num_train:num_train+num_valid]),
                   torch.tensor(items[num_train:num_train+num_valid]),
                   torch.tensor(labels[num_train:num_train+num_valid]).float(),
                   valid_user_ratings, valid_item_ratings, valid_user_sparse_ratings,
                   valid_item_sparse_ratings]

    test_user_ratings, test_item_ratings, test_user_sparse_ratings, test_item_sparse_ratings = \
        convert_records_to_rating(users[num_train+num_valid:], items[num_train+num_valid:], labels[num_train+num_valid:],
                                  num_users, num_items)
    test_queue = [torch.tensor(users[num_train+num_valid:]),
                  torch.tensor(items[num_train+num_valid:]),
                  torch.tensor(labels[num_train+num_valid:]).float(),
                  test_user_ratings, test_item_ratings, test_user_sparse_ratings,
                  test_item_sparse_ratings
                  ]
    return train_queue, valid_queue, test_queue


def get_data_queue_cf_nonsparse(data_path, args):
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'

    data_path = 'data/' + data_path

    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if args.dataset == 'ml-100k':
                line = line.split()
            elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                line = line.split('::')
            elif args.dataset == 'ml-20m':
                if i == 0:
                    continue
                line = line.split(',')
            user = int(line[0]) - 1
            item = int(line[1]) - 1
            label = float(line[2])
            users.append(user)
            items.append(item)
            labels.append(label)

    labels = StandardScaler().fit_transform(
        np.reshape(labels, [-1, 1])).flatten().tolist()

    users, items, labels = shuffle(users, items, labels)
    num_train = int(len(users) * args.train_portion)
    num_valid = int(len(users) * args.valid_portion)

    num_users = max(users) + 1
    num_items = max(items) + 1

    if not args.minibatch:
        user_ratings, item_ratings = \
            convert_records_to_rating_nonsparse(
                users[:num_train], items[:num_train], labels[:num_train], num_users, num_items)
        train_queue = [torch.tensor(users[:num_train]),
                       torch.tensor(items[:num_train]),
                       torch.tensor(labels[:num_train]),
                       user_ratings, item_ratings, user_ratings, item_ratings
                       ]
    else:
        train_queue = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(users[:num_train]), torch.tensor(items[:num_train]),
            torch.tensor(labels[:num_train])), batch_size=4096)
    valid_user_ratings, valid_item_ratings = \
        convert_records_to_rating_nonsparse(users[num_train:num_train+num_valid],
                                            items[num_train:num_train+num_valid],
                                            labels[num_train:num_train+num_valid], num_users, num_items)

    valid_queue = [torch.tensor(users[num_train:num_train+num_valid]),
                   torch.tensor(items[num_train:num_train+num_valid]),
                   torch.tensor(labels[num_train:num_train+num_valid]).float(),
                   valid_user_ratings, valid_item_ratings, valid_user_ratings,
                   valid_item_ratings]

    test_user_ratings, test_item_ratings = \
        convert_records_to_rating_nonsparse(users[num_train+num_valid:], items[num_train+num_valid:], labels[num_train+num_valid:],
                                            num_users, num_items)
    test_queue = [torch.tensor(users[num_train+num_valid:]),
                  torch.tensor(items[num_train+num_valid:]),
                  torch.tensor(labels[num_train+num_valid:]).float(),
                  test_user_ratings, test_item_ratings, test_user_ratings, test_item_ratings
                  ]
    return train_queue, valid_queue, test_queue


def convert_records_to_rating_nonsparse(users, items, labels, num_users, num_items):
    users_ratings, items_ratings = dict(), dict()
    for k in range(len(users)):
        user, item = users[k], items[k]
        if user in users_ratings:
            users_ratings[user].append(item+1)
        else:
            users_ratings[user] = [item+1]
        if item in items_ratings:
            items_ratings[item].append(user+1)
        else:
            items_ratings[item] = [user+1]
    ui_idx, ui_labels = get_index(users_ratings, labels)
    iu_idx, iu_labels = get_index(items_ratings, labels)

    users_ratings = [users_ratings[u] for u in users]
    items_ratings = [items_ratings[i] for i in items]
    max_u_rating = max([len(u_rating) for u_rating in users_ratings])
    max_i_rating = max([len(i_rating) for i_rating in items_ratings])
    users_ratings = [list(u_rating + [0]*(max_u_rating-len(u_rating)))
                     for u_rating in users_ratings]
    items_ratings = [list(i_rating + [0]*(max_i_rating-len(i_rating)))
                     for i_rating in items_ratings]

    return torch.tensor(users_ratings), torch.tensor(items_ratings)


def convert_records_to_rating(users, items, labels, num_users, num_items):
    users_ratings, items_ratings = dict(), dict()
    for k in range(len(users)):
        user, item = users[k], items[k]
        if user in users_ratings:
            users_ratings[user].append(item+1)
        else:
            users_ratings[user] = [item+1]
        if item in items_ratings:
            items_ratings[item].append(user+1)
        else:
            items_ratings[item] = [user+1]
    ui_idx, ui_labels = get_index(users_ratings, labels)
    iu_idx, iu_labels = get_index(items_ratings, labels)

    users_ratings = [users_ratings[u] for u in users]
    items_ratings = [items_ratings[i] for i in items]
    max_u_rating = max([len(u_rating) for u_rating in users_ratings])
    max_i_rating = max([len(i_rating) for i_rating in items_ratings])
    users_ratings = [list(u_rating + [0]*(max_u_rating-len(u_rating)))
                     for u_rating in users_ratings]
    items_ratings = [list(i_rating + [0]*(max_i_rating-len(i_rating)))
                     for i_rating in items_ratings]

    print(ui_idx.shape, ui_labels.shape)
    user_sparse_ratings = torch.sparse.FloatTensor(
        ui_idx, ui_labels, torch.Size([len(users), num_items])).to_dense()
    item_sparse_ratings = torch.sparse.FloatTensor(
        iu_idx, iu_labels, torch.Size([len(items), num_users])).to_dense()
    return torch.tensor(users_ratings), torch.tensor(items_ratings), user_sparse_ratings.float(), item_sparse_ratings.float()


def get_index(ratings_dict, labels):
    ui_idx_len = sum([len(ratings_dict[k])*len(ratings_dict[k])
                      if k in ratings_dict else 0 for k in range(len(ratings_dict))])
    ui_idx = np.zeros([2, ui_idx_len])
    new_labels = np.zeros([ui_idx_len])

    count, x_count = 0, 0
    for i in range(len(ratings_dict)):
        if i not in ratings_dict:
            continue
        # if i > 2:
            # break
        print(i)
        u_r = ratings_dict[i]
        for p in range(len(u_r)):
            for q in range(len(u_r)):
                ui_idx[0, count] = x_count
                ui_idx[1, count] = u_r[q] - 1
                new_labels[count] = labels[x_count]
                count += 1
            x_count += 1
    return torch.LongTensor(ui_idx), torch.tensor(new_labels).float()


def get_data_queue_efficiently(data_path, args):
    '''从数据集中获取train_queue, valid_queue, test_queue, explicit'''
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'
    elif args.dataset == 'amazon-book':
        data_path += 'ratings.dat'
    elif args.dataset == 'yelp':
        data_path += 'ratings.dat'
    elif args.dataset == 'yelp2':
        data_path += 'ratings.dat'

    elif args.dataset == 'yelp-10k':
        data_path += 'ratings-10k.dat'
    elif args.dataset == 'yelp-50k':
        data_path += 'ratings-50k.dat'
    elif args.dataset == 'yelp-100k':
        data_path += 'ratings-100k.dat'
    elif args.dataset == 'yelp-1m':
        data_path += 'ratings-1m.dat'
    elif args.dataset == 'yelp-all':
        data_path += 'ratings-all.dat'

    data_path = 'data/' + data_path

    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if args.dataset == 'ml-100k':
                line = line.split()
            elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                line = line.split('::')
            elif args.dataset == 'ml-20m':
                if i == 0:
                    continue
                line = line.split(',')
            elif args.dataset == 'amazon-book':
                line = line.split(',')
            elif args.dataset == 'yelp' or 'yelp2':
                line = line.split(',')
            user = int(line[0]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[0])
            item = int(line[1]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[1])
            label = float(line[2])
            users.append(user)
            items.append(item)
            labels.append(label)

    labels = StandardScaler().fit_transform(np.reshape(labels, [-1, 1])).flatten().tolist()

    users, items, labels = shuffle(users, items, labels)
    num_train = int(len(users) * args.train_portion)
    num_valid = int(len(users) * args.valid_portion)

    users_train = np.array(users[:num_train], dtype=np.int32)
    items_train = np.array(items[:num_train], dtype=np.int32)
    labels_train = np.array(labels[:num_train], dtype=np.float32)

    num_users = max(users) + 1
    num_items = max(items) + 1
    user_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (users_train, items_train)), shape=(num_users, num_items)).tocsr().toarray())
    item_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (items_train, users_train)), shape=(num_items, num_users)).tocsr().toarray())


    train_queue = [torch.tensor(users[:num_train]),
                   torch.tensor(items[:num_train]),
                   torch.tensor(labels[:num_train]),
                   user_interactions, item_interactions]

    valid_queue = [torch.tensor(users[num_train:num_train+num_valid]),
                   torch.tensor(items[num_train:num_train+num_valid]),
                   torch.tensor(labels[num_train:num_train+num_valid]),
                   user_interactions, item_interactions]

    test_queue = [torch.tensor(users[num_train+num_valid:]),
                  torch.tensor(items[num_train+num_valid:]),
                  torch.tensor(labels[num_train+num_valid:]),
                  user_interactions, item_interactions]

    return train_queue, valid_queue, test_queue


def get_data_queue_negsampling_efficiently(data_path, args):
    '''implicit数据集组织方法'''
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'
    elif args.dataset == 'amazon-book':
        # data_path += 'ratings2.dat'
        data_path += 'ratings_Books.csv'
    elif args.dataset == 'yelp':
        data_path += 'ratings.dat'
    elif args.dataset == 'yelp2':
        data_path += 'ratings.dat'
        
    elif args.dataset == 'yelp-10k':
        data_path += 'ratings-10k.dat'
    elif args.dataset == 'yelp-100k':
        data_path += 'ratings-100k.dat'
    elif args.dataset == 'yelp-1m':
        data_path += 'ratings-1m.dat'

    # if args.dataset == 'yelp-100k' or args.dataset == 'yelp-1m':
    #     data_path = 'data/' + 'yelp'
    # else:
    #     data_path = 'data/' + data_path
    data_path = 'data/' + data_path
    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if args.dataset == 'ml-100k':
                line = line.split()
            elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                line = line.split('::')
            elif args.dataset == 'ml-20m':
                if i == 0:
                    continue
                line = line.split(',')
            elif args.dataset == 'amazon-book':
                line = line.split(',')
            elif args.dataset == 'yelp' or 'yelp2':
                line = line.split(',')
            user = int(line[0]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[0])
            item = int(line[1]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[1])
            label = float(line[2])
            users.append(user)
            items.append(item)
            labels.append(label)

    users, items, labels = shuffle(users, items, labels)
    num_train = int(len(users) * args.train_portion)
    num_valid = int(len(users) * args.valid_portion)

    users_train = np.array(users[:num_train], dtype=np.int32)
    items_train = np.array(items[:num_train], dtype=np.int32)
    labels_train = np.array(labels[:num_train], dtype=np.float32)

    num_users = max(users) + 1
    num_items = max(items) + 1
    user_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (users_train, items_train)), shape=(num_users, num_items)).tocsr().toarray())
    item_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (items_train, users_train)), shape=(num_items, num_users)).tocsr().toarray())
    a = time.time()
    negs_train = np.zeros(len(labels_train), dtype=np.int64)
    for k in range(len(labels_train)):
        neg = np.random.randint(num_items)
        while user_interactions[users_train[k], neg] != 0:
            neg = np.random.randint(num_items)
        negs_train[k] = neg
    negs_train = torch.from_numpy(negs_train)


    train_queue_pair = [torch.tensor(users[:num_train]),
                        torch.tensor(items[:num_train]),
                        negs_train,
                        user_interactions, item_interactions]

    valid_queue = [torch.tensor(users[num_train:num_train+num_valid]),
                   torch.tensor(items[num_train:num_train+num_valid]),
                   torch.tensor(labels[num_train:num_train+num_valid]),
                   user_interactions, item_interactions]

    users_test = np.array(users[num_train+num_valid:], dtype=np.int32)
    items_test = np.array(items[num_train+num_valid:], dtype=np.int32)
    labels_test = np.array(labels[num_train+num_valid:], dtype=np.float32)
    test_user_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_test, (users_test, items_test)), shape=(num_users, num_items)).tocsr().toarray())

    a = np.argsort(users[num_train+num_valid:])
    test_queue = [torch.tensor(np.array(users[num_train+num_valid:], dtype=np.int64)[a]),
                  torch.tensor(np.array(items[num_train+num_valid:], dtype=np.int64)[a]),
                  test_user_interactions,
                  user_interactions, item_interactions]
    return train_queue_pair, valid_queue, test_queue


def get_data_queue_subsampling_efficiently(data_path, args):
    '''implicit数据集组织方法'''
    users, items, labels = [], [], []
    if args.dataset == 'ml-100k':
        data_path += 'u.data'
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
        data_path += 'ratings.dat'
    elif args.dataset == 'ml-20m':
        data_path += 'ratings.csv'
    elif args.dataset == 'amazon-book':
        # data_path += 'ratings2.dat'
        data_path += 'ratings_Books.csv'
    elif args.dataset == 'yelp':
        data_path += 'ratings.dat'
        # data_path += 'ratings_100k_num.dat'
    elif args.dataset == 'yelp2':
        data_path += 'ratings.dat'

    data_path = 'data/' + data_path

    with open(data_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if args.dataset == 'ml-100k':
                line = line.split()
            elif args.dataset == 'ml-1m' or args.dataset == 'ml-10m':
                line = line.split('::')
            elif args.dataset == 'ml-20m':
                if i == 0:
                    continue
                line = line.split(',')
            elif args.dataset == 'amazon-book':
                line = line.split(',')
            elif args.dataset == 'yelp' or 'yelp2':
                line = line.split(',')
            user = int(line[0]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[0])
            item = int(line[1]) - 1 if args.dataset != 'amazon-book' and args.dataset != 'yelp' and args.dataset != 'yelp2' else int(line[1])
            label = float(line[2])
            users.append(user)
            items.append(item)
            labels.append(label)

    users, items, labels = shuffle(users, items, labels)
    num_train = int(len(users) * args.train_portion)
    num_valid = int(len(users) * args.valid_portion)

    users_train = np.array(users[:num_train], dtype=np.int32)
    items_train = np.array(items[:num_train], dtype=np.int32)
    labels_train = np.array(labels[:num_train], dtype=np.float32)

    num_users = max(users) + 1
    num_items = max(items) + 1
    user_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (users_train, items_train)), shape=(num_users, num_items)).tocsr().toarray())
    item_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_train, (items_train, users_train)), shape=(num_items, num_users)).tocsr().toarray())
    a = time.time()
    negs_train = np.zeros(len(labels_train), dtype=np.int64)
    for k in range(len(labels_train)):
        neg = np.random.randint(num_items)
        while user_interactions[users_train[k], neg] != 0:
            neg = np.random.randint(num_items)
        negs_train[k] = neg
    negs_train = torch.from_numpy(negs_train)


    train_queue_pair = [torch.tensor(users[:num_train]),
                        torch.tensor(items[:num_train]),
                        negs_train,
                        user_interactions, item_interactions]

    valid_queue = [torch.tensor(users[num_train:num_train+num_valid]),
                   torch.tensor(items[num_train:num_train+num_valid]),
                   torch.tensor(labels[num_train:num_train+num_valid]),
                   user_interactions, item_interactions]

    users_test = np.array(users[num_train+num_valid:], dtype=np.int32)
    items_test = np.array(items[num_train+num_valid:], dtype=np.int32)
    labels_test = np.array(labels[num_train+num_valid:], dtype=np.float32)
    test_user_interactions = torch.from_numpy(sp.coo_matrix(
        (labels_test, (users_test, items_test)), shape=(num_users, num_items)).tocsr().toarray())

    a = np.argsort(users[num_train+num_valid:])
    test_queue = [torch.tensor(np.array(users[num_train+num_valid:], dtype=np.int64)[a]),
                  torch.tensor(np.array(items[num_train+num_valid:], dtype=np.int64)[a]),
                  test_user_interactions,
                  user_interactions, item_interactions]
    return train_queue_pair, valid_queue, test_queue


