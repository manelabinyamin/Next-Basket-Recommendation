# -*- coding:utf-8 -*-
import os
import logging
import torch
import numpy as np
import pandas as pd
from numba import njit
from sklearn.utils import shuffle


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def load_model_file(checkpoint_dir):
    MODEL_DIR = 'runs/' + checkpoint_dir
    names = [name for name in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, name))]
    max_epoch = 0
    choose_model = ''
    for name in names:
        if int(name[6:8]) >= max_epoch:
            max_epoch = int(name[6:8])
            choose_model = name
    MODEL_FILE = 'runs/' + checkpoint_dir + '/' + choose_model
    return MODEL_FILE


def sort_batch_of_lists(uids, baskets, reorder_baskets, neg_baskets, lens):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = np.argsort(-lens)
    uids = uids[sorted_idx]
    lens = lens[sorted_idx]
    baskets = baskets[sorted_idx]
    reorder_baskets = reorder_baskets[sorted_idx]
    neg_baskets = neg_baskets[sorted_idx]
    return uids, baskets, reorder_baskets, neg_baskets, lens


def pad_batch_of_lists(batch_of_lists, pad_len):
    """Pad batch of lists."""
    return [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]


def batch_iter(x_data, y_data, batch_size, pad_len, to_shuffle=True, config=None):
    """
    Turn dataset into iterable batches.

    Args:
        data: The data
        batch_size: The size of the data batch
        pad_len: The padding length
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data_size = len(x_data)
    num_batches_per_epoch = data_size // batch_size
    if to_shuffle:
        shuffled_x, shuffled_y = shuffle(x_data, y_data)
    else:
        shuffled_x, shuffled_y = x_data, y_data

    for i in range(num_batches_per_epoch):
        b_start, b_end = i * batch_size, (i + 1) * batch_size
        x_columns = ['user_id', 'baskets']
        y_columns = ['reorder_baskets', 'neg_baskets']
        uids,bks = list(shuffled_x[b_start:b_end][x_columns].values.transpose())
        re_bks,neg_bks = list(shuffled_y[b_start:b_end][y_columns].values.transpose())
        lens = np.array(list(map(len, bks)))
        uids, bks, re_bks, neg_bks, lens = sort_batch_of_lists(uids, bks, re_bks, neg_bks, lens)
        bks = pad_batch_of_lists(bks, pad_len)
        yield uids, bks, re_bks, neg_bks, lens

    if data_size % batch_size != 0:
        residual = [i for i in range(num_batches_per_epoch * batch_size, data_size)] + list(
            np.random.choice(data_size, batch_size - data_size % batch_size))
        x_columns = ['user_id', 'baskets']
        y_columns = ['reorder_baskets', 'neg_baskets']
        uids, bks = list(shuffled_x.iloc[residual][x_columns].values.transpose())
        re_bks, neg_bks = list(shuffled_y.iloc[residual][y_columns].values.transpose())
        lens = np.array(list(map(len, bks)))
        uids, bks, re_bks, neg_bks, lens = sort_batch_of_lists(uids, bks, re_bks, neg_bks, lens)
        bks = pad_batch_of_lists(bks, pad_len)
        yield uids, bks, re_bks, neg_bks, lens


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)


def pool_sum(tensor, dim):
    return torch.sum(tensor, dim)
