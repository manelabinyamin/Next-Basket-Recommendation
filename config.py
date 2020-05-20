# -*- coding:utf-8 -*-
class Config(object):
    def __init__(self):
        self.x_train = 'x_train_10.json'
        self.y_train = 'y_train_10.json'
        self.x_val = 'x_val_10.json'
        self.y_val = 'y_val_10.json'
        self.MODEL_DIR = 'runs/'
        self.cuda = True
        self.clip = 10
        self.epochs = 100
        self.batch_size = 256
        self.seq_len = 15
        self.learning_rate = 0.001  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'avg'  # ['avg', 'max', 'sum']
        self.rnn_type = 'GRU'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        self.dropout = 0.5
        self.num_product = 49688  # Embedding Layer
        self.none_idx = self.num_product + 1
        self.embedding_dim = 100  # Embedding Layer
        self.top_k = 5  # Top K
        self.recall_weight = 1000
        self.loss = 'Neg_samp'  # ['BPR', 'Multi_labeled', 'Neg_samp']
        self.substract_bias = True
        self.adaptive_lr = True
        self.use_neg_baskets = True
        self.neg_basket_ratio = 1.0
        self.prediction_threshold = 0.5
        self.calc_train_f1 = True
        self.l2 = 0.000
        self.encdr_decdr_regularization = 5e-4
        self.tie_weights = True
