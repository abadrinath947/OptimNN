from tqdm import tqdm 
import numpy as np

import sys
import itertools
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from preprocess import *
from sklearn.metrics import *
from torch.nn import functional as F
from collections import defaultdict
from torch import nn
import torch.optim as optim
from pyBKT.models import Model

data = pd.read_csv(sys.argv[1], encoding = 'latin')
cogtutor = 'template_id' not in data.columns
num_epochs = 6 if not cogtutor else 2
train_batch_size = 64 if not cogtutor else 256
val_batch_size = 32
train_split = 0.8

hidden_dim = 128
variant = sys.argv[2]

class BKTNN(nn.Module):

    def __init__(self, num_skills, num_multi, hidden_dim, variant):
        super(BKTNN, self).__init__()
        assert variant == 'KT-IDEM' or variant == 'ILE'
        input_dim = 32 if cogtutor else hidden_dim
        self.nn_skill_params = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                         
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                         
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                         
                                       nn.Linear(hidden_dim, 3), nn.Sigmoid()
                ).cuda()
        self.nn_multiplex_params = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                          
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                          
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),                          
                                       nn.Linear(hidden_dim, 3), nn.Sigmoid()
                ).cuda()
        self.nn_prior = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1), nn.Sigmoid()
                ).cuda()
        self.skill_embed = nn.Embedding(num_skills, 32 if cogtutor else hidden_dim)
        self.multiplex_embed = nn.Embedding(num_multi, 16 if cogtutor else hidden_dim)
        self.num_skills = num_skills
        self.variant = variant

    def forward(self, X, y):
        B, T = y.shape
        X = torch.where(X == -1, 0, X)
        
        skill_emb = self.skill_embed(X[..., 0])
        skill_params = self.nn_skill_params(skill_emb)
        if cogtutor:
            mplex_raw = self.multiplex_embed(X[..., 1])
            multiplex_params = self.nn_multiplex_params(torch.cat([skill_emb[..., :16] + mplex_raw, skill_emb[..., 16:]], dim = -1)) 
        else:
            multiplex_params = self.nn_multiplex_params(skill_emb + self.multiplex_embed(X[..., 1]))
        params = torch.zeros_like(skill_params).cuda()
        if self.variant == 'KT-IDEM':
            params[..., 0] = skill_params[..., 0]
            params[..., 1:] = multiplex_params[..., 1:]
        else:
            params[..., 0] = multiplex_params[..., 0]
            params[..., 1:] = skill_params[..., 1:]
        params = torch.clamp(params, 1e-6, 1 - 1e-6)
        latent = self.nn_prior(skill_emb[:, 0]).squeeze()

        corrects, latents = torch.zeros((B, T)).cuda(), torch.zeros((B, T)).cuda()
        for t in range(T):
            latents[:, t] = latent.squeeze()
            correct, latent = self.extract_latent_correct(params[:, t], latent, torch.where(y[:, t] == -1, 0, y[:, t]))
            corrects[:, t] = correct.squeeze()
        return corrects, latents

    def extract_latent_correct(self, params, latent, true_correct):
        l, g, s = params[..., 0], params[..., 1], params[..., 2]
        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        k_t = torch.where(true_correct > 0.5, k_t1, k_t0)
        next_latent = k_t + (1 - k_t) * l
        return correct, torch.clamp(next_latent, 1e-6, 1 - 1e-6)

def preprocess_data(data):
    features = ['correct', 'skill_id', multi_col]
    seqs = data.groupby(['user_id', 'skill_id'])[features].apply(lambda x: x.values.tolist())
    length = max(seqs.str.len()) + 1
    seqs = seqs.apply(lambda s: s + (length - min(len(s), length)) * [[-1] * len(features)])
    return seqs

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    groups = raw_data.groupby(['user_id', 'skill_id']).size()
    data = raw_data.set_index(['user_id', 'skill_id'])
    idx = 0
    if val:
        batch_size = val_batch_size
    else:
        batch_size = train_batch_size
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


def train(model, batches_train, batches_test, num_epochs):
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

        if  epoch == num_epochs - 1:
            batches_test = construct_batches(data_test, val = True)
            ypred, ytrue = evaluate(model, batches_test)
            model.eval()
            auc = roc_auc_score(ytrue, ypred)
            acc = (ytrue == ypred.round()).mean()
            rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
            print(f"Epoch {epoch}/{num_epochs} - [TEST AUC: {auc}] - [TEST ACC: {acc}] - [TEST RMSE: {rmse}]")
            torch.save(model.state_dict(), f"ckpts_optimnn/model-variant-{variant}-{epoch}-{auc}-{acc}-{rmse}.pth")

def train_test_split(data, skill_list = None):
    np.random.seed(42)
    data = data.set_index(['user_id', 'skill_id'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_test = data.loc[test_idx].reset_index()
    return data_train, data_test

def bkt_benchmark(train_data, test_data, **model_type):
    model = Model()
    model.fit(data = train_data, **model_type)
    print(model.params().reset_index().shape)
    preds = model.predict(data = test_data)
    assert len(preds) == len(test_data)
    return model.evaluate(data = test_data, metric = ['auc', 'accuracy', 'rmse'])

if __name__ == '__main__': 
    data_train, data_val, data_test = preprocess(data, True)
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name'

    #print("BKT:", bkt_benchmark(data_train, data_test))
    #print("KT-IDEM:", bkt_benchmark(data_train, data_test, multigs = True))
    #print("ILE:", bkt_benchmark(data_train, data_test, multilearn = True))

    print("Train-test split complete...")

    model = BKTNN(data_train['skill_id'].max() + 1, data_train[multi_col].max()+1, hidden_dim, variant).cuda()
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    print("Beginning training...")
    train(model, data_train, data_test, num_epochs)
