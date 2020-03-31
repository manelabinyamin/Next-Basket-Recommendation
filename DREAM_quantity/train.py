# -*- coding:utf-8 -*-
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math
import random
import time
import logging
import pickle
import torch
import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
from utils import data_helpers as dh
from DREAM.config import Config
from DREAM.rnn_model import DRModel
from torch.nn import Parameter


logging.info("DREAM Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime().replace(':','_')))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)


def calc_acc(scores, pos_idx):
    # Calculate hit-ratio
    prod = scores.index(max(scores[1:-1]))
    return prod in pos_idx


def train():

    def multi_label_loss(uids, baskets, dynamic_user, item_embedding):
        '''
        weighted multi-labeled loss function
        :param uids: batch of users' ID
        :param baskets: batch of users' baskets
        :param dynamic_user:  batch of users' dynamic representations
        :param item_embedding: item_embedding matrix
        :return: loss and accuracy
        '''
        loss = 0
        acc = 0
        acc_denom = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)

                    labels = pos_idx.unsqueeze(0)
                    target = torch.zeros(labels.size(0), Config().num_product).scatter_(1, labels, 1.)

                    # Score p(u, t, v > v')
                    scores = torch.sigmoid(du_p_product[t - 1])
                    r_w = Config().recall_weight
                    p_w = Config().precision_weight
                    a = scores.detach().numpy().min()
                    b = scores.detach().numpy().max()
                    l = -torch.mean(r_w * (target * torch.log(scores)) + p_w * ((1 - target) * torch.log(1 - scores)))
                    loss_u.append(l)

                    # calc accuracy
                    acc_scores = list(du_p_product.data.numpy()[0])
                    acc_pos = pos_idx.data.tolist()
                    acc += calc_acc(acc_scores, acc_pos)
                    acc_denom += 1.0

            for i in loss_u:
                loss = loss + i / len(loss_u)
        avg_loss = torch.div(loss, len(baskets))
        return avg_loss, float(acc / acc_denom)


    def bpr_loss(uids, baskets, dynamic_user, item_embedding, biases):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            baskets: batch of users' baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        loss = 0
        acc = 0
        acc_denom = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)

                    # Sample negative products
                    old_pos = set(list(train_data.loc[train_data.userID==uid]['all_products'])[0])
                    old_pos -= set(basket_t)
                    # neg = random.sample(list(old_pos), 1) if len(old_pos)>0 else []
                    # neg.extend(random.sample(list(neg_samples[uid]), len(basket_t)-len(neg)))
                    all_negative = list(neg_samples[uid])
                    all_negative.extend(list(old_pos))
                    neg = random.sample(all_negative, len(basket_t))
                    # neg = random.sample(list(neg_samples[uid]), len(basket_t))
                    neg_idx = torch.LongTensor(neg)

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]
                    bias_score = biases[pos_idx] - biases[neg_idx] if Config().substract_bias else 0

                    # Average Negative log likelihood for basket_t
                    loss_u.append(torch.mean(-torch.nn.LogSigmoid()(score-bias_score)))

                    # calc accuracy
                    acc_scores = list((du_p_product-biases).data.numpy()[0])
                    acc_pos = pos_idx.data.tolist()
                    acc += calc_acc(acc_scores, acc_pos)
                    acc_denom += 1.0

            for i in loss_u:
                loss = loss + i / len(loss_u)
        avg_loss = torch.div(loss, len(baskets))
        return avg_loss, float(acc/acc_denom)

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(Config().batch_size)
        train_loss = []
        train_acc = []
        start_time = time.clock()
        num_batches = ceil(len(train_data) / Config().batch_size)
        loss_function = bpr_loss if Config().loss == 'BPR' else multi_label_loss
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=True)):
            uids, baskets, lens = x
            model.zero_grad()
            dynamic_user, _ = model(baskets, lens, dr_hidden)

            loss, acc = loss_function(uids, baskets, dynamic_user, model.encode.weight, biases)
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_loss.append(loss.data)
            train_acc.append(acc)

            # Logging
            if i % Config().log_interval == 0 and i > 0:
                elapsed = (time.clock() - start_time) / Config().log_interval
                cur_loss = train_loss[-1]
                start_time = time.clock()
                logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} | Accuracy {:05.4f} |'
                            .format(epoch, i, num_batches, elapsed, cur_loss, acc))

        return np.mean(train_acc), np.mean(train_loss)


    def test_model():
        model.eval()
        item_embedding = model.encode.weight
        dr_hidden = model.init_hidden(Config().batch_size)

        hitratio_numer = 0
        hitratio_denom = 0

        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                du_latest = du[l - 1].unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                positives = test_data[test_data['userID'] == uid].baskets.values[0]  # list dim 1
                p_length = len(positives)

                scores = list((torch.mm(du_latest, item_embedding.t())-biases).data.numpy()[0])
                # Calculate hit-ratio
                highest_score = []
                for k in range(Config().top_k):
                    prod = scores.index(max(scores))
                    highest_score.append(prod)
                    scores[prod] = -9999
                hitratio_numer += len((set(positives) & set(highest_score)))
                hitratio_denom += min(Config().top_k,p_length)

        hit_ratio = hitratio_numer / hitratio_denom
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} |'
                    .format(epoch, hit_ratio))
        return hit_ratio


    # Load data
    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("Test data processing...")
    test_data = dh.load_data(Config().TESTSET_DIR)

    logger.info("Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Model config
    model = DRModel(Config())
    if Config().cuda:
        raise Exception('Add cuda biases')
    else:
        biases = Parameter(torch.tensor(np.random.uniform(-0.03,0.03, Config().num_product)), requires_grad=True)

    params = list(model.parameters())
    params.append(biases)
    biases = biases if Config().substract_bias else 0
    # Optimizer
    optimizer = torch.optim.Adam(params, lr=Config().learning_rate)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}.model'

    best_hit_ratio = None

    # build graphs
    plt.ion()
    fig = plt.figure()
    ax1 = plt.subplot(111)
    # plt.axis([0, 1000, 0, 1])
    all_train_acc = []
    all_test_acc = []
    ax1.plot(range(len(all_train_acc)), all_train_acc, color='blue', label='train')
    ax1.plot(range(len(all_test_acc)), all_test_acc, color='red', label='val')
    ax1.legend()

    try:
        # Training
        for epoch in range(Config().epochs):
            train_acc, train_loss = train_model()
            logger.info('-' * 89)

            test_acc = test_model()
            logger.info('-' * 89)

            # plot results
            plt.title('Accuracy')
            all_train_acc.append(train_acc)
            all_test_acc.append(test_acc)
            ax1.plot(range(len(all_train_acc)), all_train_acc, color='blue', label='train')
            ax1.plot(range(len(all_test_acc)), all_test_acc, color='red', label='val')
            plt.show()
            plt.pause(0.0001)

            # writer.add_scalars('Accuracy', {'train': train_acc,
            #                                'val': test_acc})
            # # writer.add_scalars('Loss', train_loss, epoch)

            # Checkpoint
            if not best_hit_ratio or test_acc > best_hit_ratio:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=test_acc), 'wb') as f:
                    torch.save(model, f)
                best_hit_ratio = test_acc
                last_best_epoch = epoch

            # adapt learning rate
            if Config().adaptive_lr and epoch-last_best_epoch>5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.2
                    print('New learning rate = ', param_group['lr'])

        plt.savefig('results/{}/run_{}.png'.format(Config().loss, timestamp), bbox_inches='tight')
        # predict

    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')


if __name__ == '__main__':
    train()
