# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import data_helpers as dh
import math
import sys


class DRModel(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, config):
        super(DRModel, self).__init__()

        # Model configuration
        self.config = config
        # Encoding
        # # random initialization
        # enc_w = torch.Tensor(config.num_product + 2, config.embedding_dim)
        # stdv = 1. / math.sqrt(enc_w.size(1))  # like in nn.Linear
        # enc_w.uniform_(-stdv, stdv)
        # pre-trained weights
        enc_w = torch.Tensor(np.load(sys.path[0]+'/embedding_layer.npy'))
        self.encode = torch.nn.Embedding(num_embeddings=config.num_product+2,
                                         embedding_dim=config.embedding_dim,
                                         padding_idx=0,
                                         _weight=enc_w)
        # self.encode.weight.requires_grad = False
        # # Decoding
        # dec_w = torch.Tensor(config.num_product + 2, config.embedding_dim)
        # stdv = 1. / math.sqrt(dec_w.size(1))  # like in nn.Linear
        # dec_w.uniform_(-stdv, stdv)
        # self.decode = torch.nn.Embedding(num_embeddings=config.num_product+2,
        #                                  embedding_dim=config.embedding_dim,
        #                                  padding_idx=0,
        #                                  _weight=dec_w)
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max, 'sum': dh.pool_sum}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim,
                                                          hidden_size=config.embedding_dim,
                                                          num_layers=config.rnn_layer_num,
                                                          batch_first=True,
                                                          dropout=config.dropout,
                                                          bidirectional=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.embedding_dim,
                                    hidden_size=config.embedding_dim,
                                    num_layers=config.rnn_layer_num,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=False)

    def forward(self, x, lengths, hidden):
        # Basket Encoding
        # users' basket sequence
        ub_seqs = []
        for user in x:  # x shape (batch of user, time_step, indice of product) nested lists
            embed_baskets = []
            for basket in user:
                try:
                    basket = torch.LongTensor(basket).resize_(1, len(basket))
                except:
                    debug=1
                basket = basket.cuda() if self.config.cuda else basket  # use cuda for acceleration
                basket = self.encode(torch.autograd.Variable(basket))  # shape: 1, len(basket), embedding_dim
                embed_baskets.append(self.pool(basket, dim=1))
            # concat current user's all baskets and append it to users' basket sequence
            ub_seqs.append(torch.cat(embed_baskets, 0).unsqueeze(0))  # shape: 1, num_basket, embedding_dim
            # Input for rnn
        ub_seqs = torch.cat(ub_seqs, 0).cuda() if self.config.cuda else torch.cat(ub_seqs, 0)  # shape: batch_size, max_len, embedding_dim        # Packed sequence as required by pytorch
        packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)

        # RNN
        output, h_u = self.rnn(packed_ub_seqs, hidden)

        # shape: [batch_size, true_len(before padding), embedding_dim]
        dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # next_basket = self.linear(dynamic_user)
        return dynamic_user, h_u

    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        self.encode.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()))
        else:
            return Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_())
