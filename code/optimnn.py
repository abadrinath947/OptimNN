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

data = pd.read_csv('data/as.csv', encoding = 'latin')
tag = 'assist' if 'template_id' in data.columns else 'cogtutor'
num_epochs = 12 if 'template_id' in data.columns else 5
batch_size = 64 if 'template_id' in data.columns else 2048
train_split = 0.8
hidden_dim, num_layers = 128, 4
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
    seqs = data.groupby(['user_id', 'skill_name'])[features].apply(lambda x: x.values.tolist())
    length = max(seqs.str.len()) + 1
    seqs = seqs.apply(lambda s: s + (length - min(len(s), length)) * [[-1] * len(features)])
    return seqs

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    groups = raw_data.groupby(['user_id', 'skill_name']).size()
    data = raw_data.set_index(['user_id', 'skill_name'])
    idx = 0
    while len(groups.iloc[idx * batch_size: (idx + 1) * batch_size].index) > 0 and idx < 600:
        if val:
            filtered_data = data.loc[groups.iloc[idx * batch_size: (idx + 1) * batch_size].index].reset_index()
        else:
            filtered_data = data.loc[groups.sample(batch_size, weights = np.log(groups + 1)).index].reset_index()
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        if batch.dtype == np.str_:
            import pdb; pdb.set_trace()
        X = torch.tensor(batch[:, 0, 1], dtype=torch.int).cuda()
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
    optimizer = optim.Adam(model.parameters(), lr = float(sys.argv[3]))
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
            torch.save(model.state_dict(), f"ckpts/model-{tag}-{hidden_dim}-{num_layers}-{epoch}-{auc}-{acc}-{rmse}.pth")

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
    print(params[:, 1])
    return (params[:, 2] > 0.5).sum() / num_skills, (params[:, 1] > 0.5).sum() / num_skills, (params[:, 0] > (1 - params[:, 2])/ params[:, 1]).sum() / num_skills

if __name__ == '__main__': 
    """
    Equation Solving Two or Fewer Steps              24253
    Percent Of                                       22931
    Addition and Subtraction Integers                22895
    Conversion of Fraction Decimals Percents         20992
    """
    data_train, data_val = preprocess(data, impute_template = False)
    print("BKT:", bkt_benchmark(data_train, data_val))

    print("Train-test split complete...")

    test_extract()

    batches_train = construct_batches(data_train)
    batches_val = construct_batches(data_val)
    model = BKTNN(len(data_train['skill_name'].unique()), hidden_dim, num_layers).cuda()
    # model.load_state_dict(torch.load('ckpts/model-cogtutor-128-4-2-0.7515170857913365-0.8599057896697341-0.3297936022281647.pth'))
    # model.load_state_dict(torch.load('ckpts/model-128-4-11-0.8198373552322507-0.7780222719315308-0.3903290629386902.pth'))
    # import pdb; pdb.set_trace()
    # test_rules(model)
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    """
    val = list(batches_val)
    for i, (X, y) in enumerate(val):
        if y.shape[0] > 5:
            corrects, latents, params, loss = model(X, y, True)
            for j in range(y.shape[1]):
                if y[:, j, :].mean() < latents[:, j, :].max() and latents[:, j, :].max() >= 0.75 and torch.unique_consecutive(y[:, j]).numel() >= 0.5 * y[:, j].numel():
                    print(i, j, y[:, j], latents[:, j])
    """
    print("Beginning training...")
    train(model, data_train, data_val, num_epochs)
