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
from utils import data_helpers as dh
from DREAM.config import Config
from DREAM.rnn_model import DRModel

logging.info("DREAM Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime().replace(':','_')))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)


def train():
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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config().learning_rate)

    def bpr_loss(uids, baskets, dynamic_user, item_embedding):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            baskets: batch of users' baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        loss = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)

                    # # Sample negative products
                    # b_neg_sample = np.delete(np.arange(start=1,stop=Config().num_product),pos_idx-1)
                    # neg = random.sample(list(b_neg_sample), len(basket_t))

                    neg = random.sample(list(neg_samples[uid]), len(basket_t))
                    neg_idx = torch.LongTensor(neg)

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]

                    # Average Negative log likelihood for basket_t
                    loss_u.append(torch.mean(-torch.nn.LogSigmoid()(score)))
            for i in loss_u:
                loss = loss + i / len(loss_u)
        avg_loss = torch.div(loss, len(baskets))
        return avg_loss

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(Config().batch_size)
        train_loss = 0
        start_time = time.clock()
        num_batches = ceil(len(train_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=True)):
            uids, baskets, lens = x
            model.zero_grad()  # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            dynamic_user, _ = model(baskets, lens, dr_hidden)

            loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_loss += loss.data

            # Logging
            if i % Config().log_interval == 0 and i > 0:
                elapsed = (time.clock() - start_time) / Config().log_interval
                cur_loss = train_loss.item() / Config().log_interval  # turn tensor into float
                train_loss = 0
                start_time = time.clock()
                logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} |'
                            .format(epoch, i, num_batches, elapsed, cur_loss))

    def validate_model():
        model.eval()
        dr_hidden = model.init_hidden(Config().batch_size)
        val_loss = 0
        start_time = time.clock()
        num_batches = ceil(len(validation_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            val_loss += loss.data

        # Logging
        elapsed = (time.clock() - start_time) * 1000 / num_batches
        val_loss = val_loss.item() / num_batches
        logger.info('[Validation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.4f} |'
                    .format(epoch, elapsed, val_loss))
        return val_loss

    def test_model():
        model.eval()
        item_embedding = model.encode.weight
        dr_hidden = model.init_hidden(Config().batch_size)

        hitratio_numer = 0
        hitratio_denom = 0
        ndcg = 0.0

        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                du_latest = du[l - 1].unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                positives = test_data[test_data['userID'] == uid].baskets.values[0]  # list dim 1
                p_length = len(positives)

                scores = list(torch.mm(du_latest, item_embedding.t()).data.numpy()[0])
                # Calculate hit-ratio
                highest_score = []
                for k in range(Config().top_k):
                    prod = scores.index(max(scores))
                    highest_score.append(prod)
                    scores[prod] = -9999
                hitratio_numer += len((set(positives) & set(highest_score)))
                hitratio_denom += min(Config().top_k,p_length)

                # Calculate NDCG
                u_dcg = 0
                u_idcg = 0
                for k in range(Config().top_k):
                    if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                        u_dcg += 1 / math.log(k + 1 + 1, 2)
                    u_idcg += 1 / math.log(k + 1 + 1, 2)
                ndcg += u_dcg / u_idcg

        hit_ratio = hitratio_numer / hitratio_denom
        # ndcg = 1
        ndcg = ndcg / len(train_data)
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} | NDCG {:05.4f} |'
                    .format(epoch, hit_ratio, ndcg))
        return hit_ratio, ndcg


    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}-{ndcg:.4f}.model'

    best_hit_ratio = None

    try:
        # Training
        for epoch in range(Config().epochs):
            train_model()
            logger.info('-' * 89)

            # val_loss = validate_model()
            # logger.info('-' * 89)

            hit_ratio, ndcg = test_model()
            logger.info('-' * 89)

            # Checkpoint
            if not best_hit_ratio or hit_ratio > best_hit_ratio:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio, ndcg=ndcg), 'wb') as f:
                    torch.save(model, f)
                best_hit_ratio = hit_ratio

        # predict

    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')


if __name__ == '__main__':
    train()
