# -*- coding:utf-8 -*-
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir)
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
import data_helpers as dh
from config import Config
from rnn_model import DRModel
from torch.nn import Parameter
from torchsummary import summary

plt.rcParams['figure.figsize'] = [10, 10]

# print('Sys path: ',sys.path[0])
# logging.info("DREAM Model Training...")
# logger = dh.logger_fn("torch-log", "kaggle/working/logs/training-{0}.log".format(time.asctime().replace(':','_')))
dilim = '-' * 120
config = Config()

if config.cuda:
    logging.info("Cuda:")
    print('Is cuda available: {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('GPU device name: {}'.format(torch.cuda.get_device_name(0)))
    dilim = '-' * 120

print(dilim)
for attr in sorted(config.__dict__):
    print('{:>50}|{:<50}'.format(attr.upper(), config.__dict__[attr]))
print(dilim)


def calc_acc(scores, pos_idx):
    # Calculate hit-ratio
    prod = scores.index(max(scores[1:-1]))
    return prod in pos_idx


def train():

    def multi_label_loss(uids, reorder_baskets, prev_reordered, neg_baskets, lens, dynamic_user, item_embedding):
        '''
        weighted multi-labeled loss function
        :param uids: batch of users' ID
        :param baskets: batch of users' baskets
        :param dynamic_user:  batch of users' dynamic representations
        :param item_embedding: item_embedding matrix
        :return: loss and accuracy
        '''
        loss = 0
        recall = []
        precision = []
        f1 = []
        for uid, re_bks, neg_bks, l, du in zip(uids, reorder_baskets, neg_baskets, lens, dynamic_user):
            du = du[l - 1].unsqueeze(0)
            scores = torch.mm(du, item_embedding.t())#.flatten()  # shape: [pad_len, num_item]
            pos_idx = torch.cuda.LongTensor(re_bks) if config.cuda else torch.LongTensor(re_bks)
            labels = pos_idx.unsqueeze(0)
            if config.cuda:
                targets = torch.zeros(labels.size(0), config.num_product+2).cuda().scatter_(1, labels, 1.)
                recall_weights = torch.ones(labels.size(0), config.num_product+2).cuda() * config.recall_weight
            else:
                targets = torch.zeros(labels.size(0), config.num_product+2).scatter_(1, labels, 1.)
                recall_weights = torch.ones(labels.size(0), config.num_product+2) * config.recall_weight
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=recall_weights)
            l = criterion(scores, targets)
            loss += l
            # pred = torch.nn.Sigmoid()(scores)
            # l2 = -torch.mean(config.recall_weight * (targets * torch.log(pred)) + ((1 - targets) * torch.log(1 - pred)))


            # Calculate accuracy, recall and f1-score
            if config.calc_train_f1:
                all_scores = torch.nn.Sigmoid()(scores.flatten()[re_bks + neg_bks]).cpu().data.numpy()
                # choose top k products
                top_k = all_scores.argsort()[-config.top_k:]
                true_predicted = (top_k < len(re_bks)).sum()
                recall.append(float(true_predicted / len(re_bks)))
                precision.append(float(true_predicted / config.top_k))
                f1.append(2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1] + 1e-6))

        avg_loss = loss / len(uids)
        avg_recall = np.mean(recall)
        avg_precision = np.mean(precision)
        avg_f1 = np.mean(f1)
        return avg_loss, avg_recall, avg_precision, avg_f1

    def bpr_loss(uids, reorder_baskets, prev_reordered, neg_baskets, lens, dynamic_user, item_embedding):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            prev_baskets: batch of users' previous baskets
            reorder_baskets: batch of users' reordered items in the baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        loss = 0
        recall = []
        precision = []
        f1 = []
        for uid, re_bks, neg_bks, l, du in zip(uids, reorder_baskets, neg_baskets, lens, dynamic_user):
            du = du[l-1].unsqueeze(0)
            du_p_product = torch.mm(du, item_embedding.t()).flatten()  # shape: [pad_len, num_item]
            pos_idx = torch.cuda.LongTensor(re_bks) if config.cuda else torch.LongTensor(re_bks)
            # Sample negative products.
            if config.use_neg_baskets:
                num_his_neg = min(int(np.ceil(len(re_bks) * config.neg_basket_ratio)), len(neg_bks))
                # choose previously ordered products
                neg = [np.random.choice(neg_bks) for _ in range(num_his_neg)]
                # choose from all products
                neg += [np.random.choice(config.num_product) for _ in range(len(re_bks) - num_his_neg)]
            else:
                # sample neg products randomly from all products
                neg = np.random.choice(config.num_product, len(re_bks))

            neg_idx = torch.cuda.LongTensor(neg) if config.cuda else torch.LongTensor(neg)
            # Score p(u, t, v > v')
            score = du_p_product[pos_idx] - du_p_product[neg_idx]
            # Average Negative log likelihood for re_basket_t
            loss += -torch.mean(torch.nn.LogSigmoid()(score))

            # Calculate accuracy, recall and f1-score
            if config.calc_train_f1:
                all_scores = torch.nn.Sigmoid()(du_p_product[re_bks + neg_bks]).cpu().data.numpy()#.flatten()
                # choose top k products
                top_k = all_scores.argsort()[-config.top_k:]
                true_predicted = (top_k < len(re_bks)).sum()
                recall.append(float(true_predicted / len(re_bks)))
                precision.append(float(true_predicted / config.top_k))
                f1.append(2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1] + 1e-6))

        avg_loss = loss / len(uids)
        avg_recall = np.mean(recall)
        avg_precision = np.mean(precision)
        avg_f1 = np.mean(f1)
        return avg_loss, avg_recall, avg_precision, avg_f1

    def bpr_loss_seq(uids, reorder_baskets, prev_reordered, neg_baskets, lens, dynamic_user, item_embedding):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            prev_baskets: batch of users' previous baskets
            reorder_baskets: batch of users' reordered items in the baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        loss = 0
        recall = []
        precision = []
        f1 = []
        ub_recall = []
        ub_precision = []
        ub_f1 = []

        for uid, re_bks, prv_re, neg_bks, ulen, du in zip(uids, reorder_baskets, prev_reordered, neg_baskets, lens, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, [re_basket_t, prv_reordered_t, neg_bks_t] in enumerate(zip(re_bks,prv_re,neg_bks)):
                if re_basket_t[0] != 0 and t != 0:
                    # positive idx
                    if re_basket_t[0]==None:
                        re_basket_t = [config.none_idx]
                    pos_idx = torch.cuda.LongTensor(re_basket_t) if config.cuda else torch.LongTensor(re_basket_t)
                    # Sample negative products.
                    if config.use_neg_baskets:
                        num_his_neg = min(int(np.ceil(len(re_basket_t)*config.neg_basket_ratio)),len(neg_bks_t))
                        # choose previously ordered products
                        neg = [np.random.choice(neg_bks_t) for _ in range(num_his_neg)]
                        # choose from all products
                        neg += [np.random.choice(config.num_product) for _ in range(len(re_basket_t)-num_his_neg)]
                    else:
                        # sample neg products randomly from all products
                        neg = np.random.choice(config.num_product, len(re_basket_t))

                    neg_idx = torch.cuda.LongTensor(neg) if config.cuda else torch.LongTensor(neg)
                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]
                    # Average Negative log likelihood for re_basket_t
                    loss_u.append(-torch.mean(torch.nn.LogSigmoid()(score)))

                    # Calculate accuracy, recall and f1-score
                    if config.calc_train_f1 and t==ulen-1:
                        # all_relevant_prods = neg_bks_t if re_basket_t[0] == config.none_idx else re_basket_t+neg_bks_t
                        all_scores = torch.nn.Sigmoid()(du_p_product[t - 1][prv_reordered_t]).cpu().data.numpy().flatten()
                        # choose top k products
                        # if all_scores.max() < config.prediction_threshold and False:  # choose an empty basket
                        #     top_k = [config.none_idx]
                        #     true_predicted = 1 if re_basket_t[0] == config.none_idx else 0
                        if len(all_scores) == 0:  # choose an empty basket
                            top_k = [config.none_idx]
                            true_predicted = 1 if re_basket_t[0] == config.none_idx else 0
                        else:
                            top_k = all_scores.argsort()[-config.top_k:]
                            target_reorder = np.where(np.isin(prv_reordered_t,re_basket_t))[0]
                            true_predicted = len(set(target_reorder)&set(top_k))
                        recall.append(float(true_predicted / len(re_basket_t)))
                        precision.append(float(true_predicted / len(top_k)))
                        f1.append(2*(recall[-1]*precision[-1])/(recall[-1]+precision[-1]+1e-6))
                        # check best possible results (upper bounds)
                        ub_true_predicted = len(set(prv_reordered_t)&set(re_basket_t))
                        ub_recall.append(float(ub_true_predicted / len(re_basket_t)))
                        ub_precision.append(1 if true_predicted>0 else 0)
                        ub_f1.append(2*(ub_recall[-1]*ub_precision[-1])/(ub_recall[-1]+ub_precision[-1]+1e-6))

            loss += torch.mean(torch.stack(loss_u))
        avg_loss = loss / len(uids)
        avg_recall = np.mean(recall)
        avg_precision = np.mean(precision)
        avg_f1 = np.mean(f1)
        avg_ub_recall, avg_ub_precision, avg_ub_f1 = np.mean(ub_recall),np.mean(ub_precision),np.mean(ub_f1)
        return avg_loss, avg_recall, avg_precision, avg_f1, avg_ub_recall, avg_ub_precision, avg_ub_f1

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(config.batch_size)
        # performances measures
        train_loss = []
        train_recall,train_precision,train_f1 = [],[],[]
        train_ub_recall,train_ub_precision,train_ub_f1 = [],[],[]
        # training
        start_time = time.process_time()
        num_batches = ceil(len(x_data) / config.batch_size)
        l_functions = {'BPR':bpr_loss, 'BPR_seq':bpr_loss_seq, 'Multi_labeled':multi_label_loss}
        loss_function = l_functions[config.loss]
        for i, x in enumerate(dh.batch_iter(x_data, config.batch_size, config.seq_len, to_shuffle=True)):
            uids, baskets, dow, hour_of_day, days2next, reorder_baskets, neg_baskets, prv_reorder, lens = x
            model.zero_grad()
            dynamic_user, _ = model(baskets, dow, hour_of_day, days2next, lens, dr_hidden)

            loss, recall, precision, f1, ub_recall, ub_precision, ub_f1 = loss_function(uids, reorder_baskets, prv_reorder, neg_baskets, lens, dynamic_user, model.decode.weight)
            # tie_encoder_decoder_loss = torch.dist(model.encode.weight, model.decode.weight)
            # loss += config.encdr_decdr_regularization * tie_encoder_decoder_loss
            # s = time.time()
            loss.backward()
            # print('backpropogation time:', time.time() - s)


            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            # Parameter updating
            optimizer.step()

            # save results
            train_recall.append(recall)
            train_precision.append(precision)
            train_f1.append(f1)
            train_loss.append(loss.item())
            train_ub_recall.append(ub_recall)
            train_ub_precision.append(ub_precision)
            train_ub_f1.append(ub_f1)

            # Logging
            if i % config.log_interval == 0 and i > 0:
                elapsed = (time.time() - start_time) / config.log_interval
                start_time = time.time()
                print('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | sec/batch {:02.2f} | Loss {:05.4f} | Recall {:05.4f} | Precision {:05.4f} | f1 {:05.4f} |'
                            .format(epoch, i, num_batches, elapsed, loss, recall, precision, f1))

        return np.mean(train_loss), np.mean(train_precision), np.mean(train_recall), np.mean(train_f1), np.mean(train_ub_recall), np.mean(train_ub_precision), np.mean(train_ub_f1)


    def evaluate_model():
        model.eval()
        item_embedding = model.decode.weight
        dr_hidden = model.init_hidden(config.batch_size)

        precision = []
        recall = []
        f1 = []
        for x in dh.batch_iter_eval(x_val, y_val, config.batch_size, config.seq_len, to_shuffle=False):
            uids, baskets, dow, hour_of_day, days2next, reorder_baskets, neg_baskets, prv_reorder, lens = x
            dynamic_user, _ = model(baskets, dow, hour_of_day, days2next, lens, dr_hidden)
            for uid, re_bks, prv_re, neg_bks, l, du in zip(uids, reorder_baskets, prv_reorder, neg_baskets, lens, dynamic_user):
                du_latest = du[l - 1].unsqueeze(0)

                # all_relevant_prods = neg_bks if re_bks[0] == config.none_idx else re_bks + neg_bks
                all_scores = torch.nn.Sigmoid()(torch.mm(du_latest, item_embedding.t())).cpu().data.numpy().flatten()[prv_re]
                # choose top k products
                # if all_scores.max() < config.prediction_threshold:  # choose an empty basket
                #     top_k = [config.none_idx]
                #     true_predicted = 1 if re_bks[0] == config.none_idx else 0
                if len(all_scores) == 0:  # choose an empty basket
                    top_k = [config.none_idx]
                    true_predicted = 1 if re_bks[0] == config.none_idx else 0
                else:
                    top_k = all_scores.argsort()[-config.top_k:]
                    target_reorder = np.where(np.isin(prv_re, re_bks))[0]
                    true_predicted = len(set(target_reorder) & set(top_k))
                recall.append(float(true_predicted / len(re_bks)))
                precision.append(float(true_predicted / len(top_k)))
                f1.append(2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1] + 1e-6))

        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        return avg_precision, avg_recall, avg_f1

    def plot_line(ax, train_l, test_l, title, add_legend=False, upper_bound=None):
        ax.plot(range(len(train_l)), train_l, color='blue', label='train')
        ax.plot(range(len(test_l)), test_l, color='red', label='val')
        if upper_bound is not None:
            ax.plot(range(len(upper_bound)), upper_bound, color='green', label='Upper-Bound')
        ax.set_title(title)
        if add_legend:
            ax.legend()

    # Load data
    print("Loading data...")
    x_train = pd.read_json(sys.path[0]+'/{}'.format(config.x_train), orient='index')  # samp_x_train.json
    y_train = pd.read_json(sys.path[0]+'/{}'.format(config.y_train), orient='index')  # samp_x_train.json
    x_val = pd.read_json(sys.path[0]+'/{}'.format(config.x_val), orient='index')
    y_val = pd.read_json(sys.path[0]+'/{}'.format(config.y_val), orient='index')

    x_data = pd.concat([x_train, x_val]).reset_index(drop=True)

    # Model config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Check if the model uses cuda...')
    model = DRModel(config)
    print('Is cuda available for the model: {}'.format(next(model.parameters()).is_cuda))
    if config.cuda:
        print('Enable cuda for model...')
        model.to(device)
        print('Is cuda available for the model: {}'.format(next(model.parameters()).is_cuda))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}.model'

    best_f1_ratio = None
    last_best_epoch = None
    # build graphs

    all_train_precision, all_train_recall, all_train_f1 = [], [], []
    all_test_precision, all_test_recall, all_test_f1 = [], [], []
    all_ub_recall, all_ub_precision, all_ub_f1 = [], [], []

    try:
        fig, axes = plt.subplots(3, 1)
        # plt.ion()
        plot_line(axes[0], [], [], 'Precision', add_legend=True, upper_bound=None)
        plot_line(axes[1], [], [], 'Recall', add_legend=True, upper_bound=None)
        plot_line(axes[2], [], [], 'F1-score', add_legend=True, upper_bound=None)
        # Training
        for epoch in range(config.epochs):
            # test_precision, test_recall, test_f1 = evaluate_model()
            # print('-' * 89)

            train_loss, train_precision, train_recall, train_f1, ub_recall, ub_precision, ub_f1 = train_model()
            test_precision, test_recall, test_f1 = evaluate_model()
            # print results
            print('-' * 89)
            print('[Upper-Bounds]| Epochs {:3d} | Recall {:02.4f} | Precision {:02.4f} | F1-Score {:02.4f} |'
                  .format(epoch, ub_recall, ub_precision, ub_f1))
            print('-' * 89)
            print('[Train]| Epochs {:3d} | Recall {:02.4f} | Precision {:02.4f} | F1-Score {:02.4f} |'
                  .format(epoch, train_recall, train_precision, train_f1))
            print('-' * 89)
            print('[Test]| Epochs {:3d} | Recall {:02.4f} | Precision {:02.4f} | F1-Score {:02.4f} |'
                  .format(epoch, test_recall, test_precision, test_f1))
            print('-' * 89)

            # save results
            all_train_precision.append(train_precision)
            all_train_recall.append(train_recall)
            all_train_f1.append(train_f1)
            all_test_precision.append(test_precision)
            all_test_recall.append(test_recall)
            all_test_f1.append(test_f1)
            all_ub_recall.append(ub_recall)
            all_ub_precision.append(ub_precision)
            all_ub_f1.append(ub_f1)
            # plot results
            # plt.close()
            # fig, axes = plt.subplots(3, 1)
            plot_line(axes[0], all_train_precision, all_test_precision, 'Precision', upper_bound=None)
            plot_line(axes[1], all_train_recall, all_test_recall, 'Recall', upper_bound=None)
            plot_line(axes[2], all_train_f1, all_test_f1, 'F1-score', upper_bound=None)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.show()
            plt.pause(0.0001)

            # writer.add_scalars('Accuracy', {'train': train_acc,
            #                                'val': test_acc})
            # # writer.add_scalars('Loss', train_loss, epoch)

            # Checkpoint
            if not best_f1_ratio or test_f1 > best_f1_ratio:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=test_f1), 'wb') as f:
                    torch.save(model, f)
                best_f1_ratio = test_f1
                last_best_epoch = epoch

            # adapt learning rate
            if config.adaptive_lr and epoch-last_best_epoch>5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001
                    print('New learning rate = ', param_group['lr'])

        plt.savefig('results/{}/run_{}.png'.format(config.loss, timestamp), bbox_inches='tight')
        # predict

    except KeyboardInterrupt:
        print('*' * 89)
        print('Early Stopping!')


if __name__ == '__main__':
    train()
