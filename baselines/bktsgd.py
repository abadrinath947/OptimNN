from tqdm import tqdm 
import numpy as np

import sys
import itertools
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
from pyBKT.models import Model
from preprocess import *

data = pd.read_csv('data/processed_algebra.csv', encoding = 'latin')
num_epochs = 5 if 'template_id' in data.columns else 2
batch_size = 64 if 'template_id' in data.columns else 512
train_split = 0.8
variant = sys.argv[1]

class BKTNN(nn.Module):

    def __init__(self, num_params):
        super(BKTNN, self).__init__()
        self.learn = nn.Embedding(num_params[0], 1)
        self.guess = nn.Embedding(num_params[1], 1)
        self.slip = nn.Embedding(num_params[2], 1)
        self.prior = nn.Embedding(num_params[3], 1)
        self.num_skills = num_skills

    def forward(self, X, y):
        B, T = y.shape

        l, g, s, p = torch.sigmoid(self.learn(X[..., 0])), torch.sigmoid(self.guess(X[..., 1])), torch.sigmoid(self.slip(X[..., 2])), torch.sigmoid(self.prior(X[..., 3]))
        params = [l.squeeze(), g.squeeze(), s.squeeze()]
        latent = p.squeeze()
        corrects, latents = torch.zeros((B, T)).cuda(), torch.zeros((B, T)).cuda()
        for t in range(T):
            latents[:, t] = latent.squeeze()
            correct, latent = self.extract_latent_correct(params, latent, torch.where(y[:, t] == -1, 0, y[:, t]))
            corrects[:, t] = correct.squeeze()
        return corrects, latents

    def extract_latent_correct(self, params, latent, true_correct):
        l, g, s = params #params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        k_t = torch.where(true_correct > 0.5, k_t1, k_t0)
        next_latent = k_t * (1 - 0) + (1 - k_t) * l
        return correct, torch.clamp(next_latent, 1e-6, 1 - 1e-6)

def preprocess_data(data):
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
        X = torch.tensor(batch[:, 0, 1:], dtype=torch.int).cuda()
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
    optimizer = optim.Adam(model.parameters(), lr = 0.1)
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
            torch.save(model.state_dict(), f"ckpts/model-puresgd-{variant}-{epoch}-{auc}-{acc}-{rmse}.pth")

def train_test_split(data, skill_list = None):
    np.random.seed(42)
    data = data.set_index(['user_id', 'skill_name'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()
    return data_train, data_val

def bkt_benchmark(train_data, test_data, **model_type):
    model = Model()
    model.fit(data = train_data, **model_type)
    print(model.params().reset_index().shape)
    preds = model.predict(data = test_data)
    assert len(preds) == len(test_data)
    return model.evaluate(data = test_data, metric = ['auc', 'accuracy', 'rmse'])

if __name__ == '__main__': 
    """
    Equation Solving Two or Fewer Steps              24253
    Percent Of                                       22931
    Addition and Subtraction Integers                22895
    Conversion of Fraction Decimals Percents         20992
    """
    data_train, data_val = preprocess(data, True)
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name' 
    data_train['multi_id'] = data_train['skill_id'].astype(str) + '-' + data_train[multi_col].astype(str)
    data_val['multi_id'] = data_val['skill_id'].astype(str) + '-' + data_val[multi_col].astype(str)

    multi_col = 'multi_id'

    train_templates = data_train[multi_col].unique()
    template_dict = {tn: i for i, tn in enumerate(train_templates)}
    print("Imputing templates...")
    repl = template_dict[data_train[multi_col].value_counts().index[0]]
    for temp_id in set(data_val[multi_col].unique()) - set(template_dict):
        template_dict[temp_id] = repl

    print("Replacing templates...")
    data_train[multi_col] = data_train[multi_col].apply(lambda s: template_dict[s])
    data_val[multi_col] = data_val[multi_col].apply(lambda s: template_dict[s])

    print("Train-test split complete...")

    num_skills = len(data_train['skill_id'].unique())
    num_multi = len(data_train['multi_id'].unique())
    if variant == 'KT-IDEM':
        params = [num_skills, num_multi, num_multi, num_skills]
        features = ['correct', 'skill_id', 'multi_id', 'multi_id', 'skill_id'] 
    elif variant == 'ILE':
        params = [num_multi,num_skills,num_skills,num_skills]
        features = ['correct', 'multi_id'] + ['skill_id'] * 3
    else:
        params = [num_skills] * 4
        features = ['correct'] + ['skill_id'] * 4

    model = BKTNN(params).cuda()
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    print("Beginning training...")
    train(model, data_train, data_val, num_epochs)
