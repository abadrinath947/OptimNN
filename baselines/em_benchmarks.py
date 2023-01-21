from tqdm import tqdm 
import numpy as np

import sys
import itertools
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from scipy.stats import t
from torch.nn import functional as F
from collections import defaultdict
from torch import nn
import torch.optim as optim
from time import time
from preprocess import *
from pyBKT.models import Model

data = pd.read_csv('data/as.csv', encoding = 'latin')

def preprocess_data(data):
    features = ['correct', 'skill_id', 'template_id']
    seqs = data.groupby(['user_id', 'skill_name'])[features].apply(lambda x: x.values.tolist())
    length = max(seqs.str.len()) + 1
    seqs = seqs.apply(lambda s: s + (length - min(len(s), length)) * [[-1] * len(features)])
    return seqs

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    groups = raw_data.groupby(['user_id', 'skill_name']).size()
    data = raw_data.set_index(['user_id', 'skill_name'])
    idx = 0
    while len(groups.iloc[idx * batch_size: (idx + 1) * batch_size].index) > 0:
        if val:
            filtered_data = data.loc[groups.iloc[idx * batch_size: (idx + 1) * batch_size].index].reset_index()
        else:
            filtered_data = data.loc[groups.sample(batch_size, weights = np.log(groups + 1)).index].reset_index()
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        X = torch.tensor(batch[:, ..., 1:], dtype=torch.int).cuda()
        y = torch.tensor(batch[:, ..., 0], dtype=torch.float32).cuda()
        yield X, y
        idx += 1

def evaluate(model, batches):
    ypred, ytrue = [], []
    for X, y in batches:
        mask = y != -1
        c, l = model(X, y)
        ypred.append(c[mask].ravel().detach().cpu().numpy())
        ytrue.append(y[mask].ravel().detach().cpu().numpy())
    ypred = np.concatenate(ypred)
    ytrue = np.concatenate(ytrue)
    return ypred, ytrue #roc_auc_score(ytrue, ypred)


def train(model, batches_train, batches_val, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    for epoch in range(num_epochs):
        model.train()
        batches_train = construct_batches(data_train)
        losses = []
        pbar = tqdm(batches_train)
        for X, y in pbar:
            optimizer.zero_grad()
            c, l = model(X, y)
            mask = (y != -1)
            if c[mask].min() < 0 or c[mask].max() > 1:
                import pdb; pdb.set_trace()
            loss = F.binary_cross_entropy(c[mask], y[mask])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f"Training Loss: {np.mean(losses)}")

        if  epoch % 1 == 0:
            batches_val = construct_batches(data_val, val = True)
            ypred, ytrue = evaluate(model, batches_val)
            model.eval()
            auc = roc_auc_score(ytrue, ypred)
            acc = (ytrue == ypred.round()).mean()
            rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
            print(f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]")
            torch.save(model.state_dict(), f"ckpts/model-mlfgs-{tag}-{epoch}-{auc}-{acc}-{rmse}.pth")

def train_test_split(data, skill_list = None):
    np.random.seed(42)
    data = data.set_index(['user_id', 'skill_name'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()
    return data_train, data_val

def bkt_benchmark(train_data, test_data, K = 5, **model_type):
    evals = []
    for k in range(K):
        model = Model()
        # model.coef_ = {skill_name: {'guesses': np.array([0.1]), 'slips': np.array([0.1])} for skill_name in data_train['skill_name'].unique()}
        model.fit(data = train_data, **model_type)
        print(model.params().reset_index().shape)
        # preds = model.predict(data = test_data)
        # assert len(preds) == len(test_data)
        evals.append(model.evaluate(data = test_data, metric = ['auc', 'accuracy', 'rmse']))
    t1, t2 = t.interval(0.9, K, np.mean(evals, axis = 0), np.std(evals, axis = 0))
    return (t2 + t1) / 2, (t2 - t1) / 2 

if __name__ == '__main__': 
    """
    Equation Solving Two or Fewer Steps              24253
    Percent Of                                       22931
    Addition and Subtraction Integers                22895
    Conversion of Fraction Decimals Percents         20992
    """
    data_train, data_val = preprocess(data, True)
    a = time()
    print("BKT:", bkt_benchmark(data_train, data_val))
    b = time()
    print("KT-IDEM:", bkt_benchmark(data_train, data_val, multigs = True))
    c = time()
    print("ILE:", bkt_benchmark(data_train, data_val, multilearn = True))
    d = time()
    print(b - a, c - b, d - c)
