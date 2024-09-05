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
from preprocess import *
from pyBKT.models import Model

data = pd.read_csv(sys.argv[3], encoding = 'latin')
tag = sys.argv[3].replace('/', '_')
num_epochs = 12 if 'template_id' in data.columns else 5
batch_size = 64 if 'template_id' in data.columns else 2048
train_split = 0.8
hidden_dim, num_layers = int(sys.argv[1]), int(sys.argv[2])
assert hidden_dim >= 16
assert 1 <= num_layers <= 32

class BKTNN(nn.Module):

    def __init__(self, num_skills, hidden_dim, num_layers):
        super(BKTNN, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, 4), nn.Sigmoid()])
        self.nn_params = nn.Sequential(*layers).cuda()
        self.skill_embed = nn.Embedding(num_skills, hidden_dim)
        self.num_skills = num_skills

    def forward(self, X, y):
        B, T = y.shape
        if X.max() >= self.num_skills or X.min() < 0 or torch.isnan(X).any():
            import pdb; pdb.set_trace()
        params = torch.clamp(self.nn_params(self.skill_embed(X)), 1e-6, 1 - 1e-6)
        latent = params[..., -1]
        corrects, latents = torch.zeros((B, T)).cuda(), torch.zeros((B, T)).cuda()
        for t in range(T):
            latents[:, t] = latent.squeeze()
            correct, latent = self.extract_latent_correct(params, latent, torch.where(y[:, t] == -1, 0, y[:, t]))
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
    features = ['correct', 'skill_id']
    seqs = data.groupby(['user_id', 'skill_id'])[features].apply(lambda x: x.values.tolist())
    length = max(seqs.str.len()) + 1
    seqs = seqs.apply(lambda s: s + (length - min(len(s), length)) * [[-1] * len(features)])
    return seqs

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    groups = raw_data.groupby(['user_id', 'skill_id']).size()
    data = raw_data.set_index(['user_id', 'skill_id'])
    idx = 0
    while len(groups.iloc[idx * batch_size: (idx + 1) * batch_size].index) > 0 and idx < 600:
        if val:
            filtered_data = data.loc[groups.iloc[idx * batch_size: (idx + 1) * batch_size].index].reset_index()
        else:
            filtered_data = data.loc[groups.sample(batch_size, weights = np.log(groups + 1)).index].reset_index()
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        if batch.dtype == np.str_:
            raise ValueError("found invalid type")
        X = torch.tensor(batch[:, 0, 1], dtype=torch.int).cuda()
        y = torch.tensor(batch[:, ..., 0], dtype=torch.float32).cuda()
        yield X, y
        idx += 1

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    #groups = raw_data.groupby(['user_id', 'skill_id']).size()
    #data = raw_data.set_index(['user_id', 'skill_id'])
    idx = 0
    import polars as pl
    df2 = pl.from_pandas(raw_data)
    grouped = df2[["user_id", "skill_id", "correct"]].groupby(["user_id", "skill_id"]).agg([pl.col("correct").explode()])
    if not val:
        grouped = grouped.sample(len(grouped))
        lens = np.log(grouped['correct'].list.lengths().to_numpy() + 1)

    while len(grouped[idx * batch_size: (idx + 1) * batch_size]) > 0:
        if not val:
            sample_idx = np.random.choice(range(len(grouped)), size=batch_size, p = lens/lens.sum(), replace=False)
            filtered_data = grouped[sample_idx, 1:].rows()
        else:
            filtered_data = grouped[idx * batch_size: (idx + 1) * batch_size, 1:].rows()
        batch = [(j, [i] * len(j)) for i, j in filtered_data]
        from text2array import Batch
        batch = np.transpose(Batch([{'x': batch}]).to_array(pad_with=-1)['x'], (0, 1, 3, 2))[0]

        batch = torch.from_numpy(batch).cuda()
        # batch_preprocessed = preprocess_data(filtered_data)
        # batch = np.array(batch_preprocessed.to_list())
        X = batch[:, 0, 1]
        y = batch[:, ..., 0].to(torch.float32)
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
    optimizer = optim.Adam(model.parameters(), lr = 5e-4)
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
            torch.save(model.state_dict(), f"ckpts_optimnn/model-{tag}-{hidden_dim}-{num_layers}-{epoch}-{auc}-{acc}-{rmse}.pth")

def train_test_split(data, skill_list = None):
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

def test_extract():
    params = torch.Tensor([[0.06411, 0.15405, 0.28270]])
    latent = torch.Tensor([[0.52355]])
    c, nl = BKTNN.extract_latent_correct(None, params, latent, torch.Tensor([[1]]))
    assert 0.448 <= c.item() <= 0.449
    assert 0.846 <= nl.item() <= 0.847
    print("Successfully tested our BKT forward update")

def test_rules(model):
    num_skills = len(data_train['skill_id'].unique())
    skills = torch.Tensor(list(range(num_skills)))
    skills = skills.long().cuda()
    params = model.nn_params(model.skill_embed(skills))
    return (params[:, 2] > 0.5).sum() / num_skills, (params[:, 1] > 0.5).sum() / num_skills, (params[:, 0] > (1 - params[:, 2])/ params[:, 1]).sum() / num_skills

if __name__ == '__main__': 
    data_train, data_val, data_test = preprocess(data, impute_template = False)
    #print("BKT:", bkt_benchmark(data_train, data_test))

    print("Train-test split complete...")

    test_extract()

    batches_train = construct_batches(data_train)
    batches_test = construct_batches(data_test)
    model = BKTNN(max(data_train['skill_id'].max(), data_test['skill_id'].max()) + 1, hidden_dim, num_layers).cuda()
    # model.load_state_dict(torch.load('ckpts/model-cogtutor-128-4-2-0.7515170857913365-0.8599057896697341-0.3297936022281647.pth'))
    # model.load_state_dict(torch.load('ckpts/model-128-4-11-0.8198373552322507-0.7780222719315308-0.3903290629386902.pth'))
    # import pdb; pdb.set_trace()
    # test_rules(model)
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    print("Beginning training...")
    train(model, data_train, data_test, num_epochs)
