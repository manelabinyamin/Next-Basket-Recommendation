# -*- coding:utf-8 -*-
class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = '../data/train.json'
        self.TESTSET_DIR = '../data/test.json'
        self.NEG_SAMPLES = '../data/all_neg2.pickle'
        self.MODEL_DIR = 'runs/'
        self.cuda = False
        self.clip = 10
        self.epochs = 200
        self.batch_size = 256
        self.seq_len = 15
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'avg'  # ['avg', 'max']
        self.rnn_type = 'LSTM'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        self.dropout = 0.5
        self.num_product = 4449+1  # 26991+1 # Embedding Layer
        self.embedding_dim = 32  # Embedding Layer
        self.top_k = 1  # Top K
        self.recall_weight = 500
        self.precision_weight = 1
