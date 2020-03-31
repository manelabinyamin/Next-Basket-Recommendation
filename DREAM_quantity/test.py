# -*- coding:utf-8 -*-
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import time
import random
import math
import pickle
import torch
import numpy as np
from DREAM.config import Config
from utils import data_helpers as dh


logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime().replace(':','_')))

MODEL = input("Please input the model file you want to test: ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")

MODEL_DIR = dh.load_model_file(MODEL)


def test():
    # Load data
    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    # Load model
    dr_model = torch.load(MODEL_DIR)

    dr_model.eval()

    item_embedding = dr_model.encode.weight
    hidden = dr_model.init_hidden(Config().batch_size)
    prediction = {}

    for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
        uids, baskets, lens = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du in zip(uids, lens, dynamic_user):
            du_latest = du[l - 1].unsqueeze(0)

            all_scores = list(torch.mm(du_latest, item_embedding.t()).data.numpy()[0])
            prod = all_scores.index(max(all_scores))
            prediction[uid] = prod

    with open('prediction.pickle', 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test()


