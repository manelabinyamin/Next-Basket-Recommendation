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

    def multi_label_loss(uids, reorder_baskets, neg_baskets, dynamic_user, item_embedding):
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
        for uid, re_bks, du in zip(uids, reorder_baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, basket_t in enumerate(re_bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)

                    labels = pos_idx.unsqueeze(0)
                    target = torch.zeros(labels.size(0), config.num_product).scatter_(1, labels, 1.)

                    # Score p(u, t, v > v')
                    scores = torch.nn.Sigmoid()(du_p_product[t - 1])
                    r_w = config.recall_weight
                    p_w = config.precision_weight
                    l = -torch.mean(r_w * (target * torch.log(scores)) + p_w * ((1 - target) * torch.log(1 - scores)))
                    loss_u.append(l)

                    # calc accuracy
                    acc_scores = list(du_p_product.data.numpy()[0])
                    acc_pos = pos_idx.data.tolist()
                    acc += calc_acc(acc_scores, acc_pos)
                    acc_denom += 1.0

            for i in loss_u:
                loss = loss + i / len(loss_u)
        return loss, float(acc / acc_denom)

    def bpr_loss(uids, reorder_baskets, neg_baskets, dynamic_user, item_embedding):
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
        for uid, re_bks, neg_bks, du in zip(uids, reorder_baskets, neg_baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, [re_basket_t, neg_bks_t] in enumerate(zip(re_bks,neg_bks)):
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
                    if config.calc_train_f1:
                        all_scores = torch.nn.Sigmoid()(du_p_product[t - 1][re_basket_t+neg_bks_t]).cpu().data.numpy().flatten()
                        # choose top k products
                        top_k = all_scores.argsort()[-config.top_k:]
                        true_predicted = (top_k < len(re_basket_t)).sum()
                        recall.append(float(true_predicted / len(re_basket_t)))
                        precision.append(float(true_predicted / config.top_k))
                        f1.append(2*(recall[-1]*precision[-1])/(recall[-1]+precision[-1]+1e-6))


            loss += torch.mean(torch.stack(loss_u))
        avg_loss = loss / len(uids)
        avg_recall = np.mean(recall)
        avg_precision = np.mean(precision)
        avg_f1 = np.mean(f1)
        return avg_loss, avg_recall, avg_precision, avg_f1

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(config.batch_size)
        train_recall = []
        train_precision = []
        train_f1 = []
        train_loss = []
        start_time = time.process_time()
        num_batches = ceil(len(x_train) / config.batch_size)
        loss_function = bpr_loss if config.loss == 'BPR' else multi_label_loss
        for i, x in enumerate(dh.batch_iter(x_train, config.batch_size, config.seq_len, shuffle=True, config=config)):
            uids, baskets, reorder_baskets, neg_baskets, lens = x
            model.zero_grad()
            dynamic_user, _ = model(baskets, lens, dr_hidden)

            # loss, recall, precision, f1 = loss_function(uids, reorder_baskets, neg_baskets, dynamic_user, model.encode.weight)

            loss, recall, precision, f1 = loss_function(uids, reorder_baskets, neg_baskets, dynamic_user, model.encode.weight)
            # tie_encoder_decoder_loss = torch.dist(model.encode.weight, model.decode.weight)
            # loss += config.encdr_decdr_regularization * tie_encoder_decoder_loss
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            # Parameter updating
            optimizer.step()

            # save results
            train_recall.append(recall)
            train_precision.append(precision)
            train_f1.append(f1)
            train_loss.append(loss.item())
            # Logging
            if i % config.log_interval == 0 and i > 0:
                elapsed = (time.time() - start_time) / config.log_interval
                start_time = time.time()
                print('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | sec/batch {:02.2f} | Loss {:05.4f} | Recall {:05.4f} | Precision {:05.4f} | f1 {:05.4f} |'
                            .format(epoch, i, num_batches, elapsed, loss, recall, precision, f1))

        return np.mean(train_loss), np.mean(train_precision), np.mean(train_recall), np.mean(train_f1)


    def evaluate_model():
        model.eval()
        item_embedding = model.encode.weight
        dr_hidden = model.init_hidden(config.batch_size)

        precision = []
        recall = []
        f1 = []

        for x in dh.batch_iter(x_val, config.batch_size, config.seq_len, shuffle=False):
            uids, baskets, _, _, lens = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                du_latest = du[l - 1].unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                pos = y_val.loc[uid].reorder_baskets  # list dim 1
                neg = y_val.loc[uid].neg_baskets

                all_scores = torch.nn.Sigmoid()(torch.mm(du_latest, item_embedding.t())).cpu().data.numpy().flatten()[pos + neg]
                # choose top k products
                top_k = all_scores.argsort()[-config.top_k:]
                true_predicted = (top_k < len(pos)).sum()
                recall.append(float(true_predicted / len(pos)))
                precision.append(float(true_predicted / config.top_k))
                f1.append(2 * (recall[-1] * precision[-1]) / (recall[-1] + precision[-1] + 1e-6))

        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        print('[Test]| Epochs {:3d} | Recall {:02.4f} | Precision {:02.4f} | F1-Score {:02.4f} |'
                    .format(epoch, avg_recall, avg_precision, avg_f1))
        return avg_precision, avg_recall, avg_f1

    def plot_line(ax, train_l, test_l, title, add_legend=False):
        ax.plot(range(len(train_l)), train_l, color='blue', label='train')
        ax.plot(range(len(test_l)), test_l, color='red', label='val')
        ax.set_title(title)
        if add_legend:
            ax.legend()

    # Load data
    print("Loading data...")
    x_train = pd.read_json(sys.path[0]+'/{}'.format(config.x_train), orient='index')  # samp_x_train.json
    x_val = pd.read_json(sys.path[0]+'/{}'.format(config.x_val), orient='index')
    y_val = pd.read_json(sys.path[0]+'/{}'.format(config.y_val), orient='index')

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

    try:
        fig, axes = plt.subplots(3, 1)
        # plt.ion()
        plot_line(axes[0], [], [], 'Precision', add_legend=True)
        plot_line(axes[1], [], [], 'Recall', add_legend=True)
        plot_line(axes[2], [], [], 'F1-score', add_legend=True)
        # Training
        for epoch in range(config.epochs):
            # test_precision, test_recall, test_f1 = evaluate_model()
            # print('-' * 89)

            train_loss, train_precision, train_recall, train_f1 = train_model()
            print('-' * 89)

            test_precision, test_recall, test_f1 = evaluate_model()
            print('-' * 89)

            # save results
            all_train_precision.append(train_precision)
            all_train_recall.append(train_recall)
            all_train_f1.append(train_f1)
            all_test_precision.append(test_precision)
            all_test_recall.append(test_recall)
            all_test_f1.append(test_f1)
            # plot results
            # plt.close()
            # fig, axes = plt.subplots(3, 1)
            plot_line(axes[0], all_train_precision, all_test_precision, 'Precision')
            plot_line(axes[1], all_train_recall, all_test_recall, 'Recall')
            plot_line(axes[2], all_train_f1, all_test_f1, 'F1-score')
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
