import numpy as np
import sys
import itertools
import datetime
import pandas as pd
import torch
from pyBKT.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from torch import nn
from dataset import *
import torch.optim as optim
from transformer import *
from preprocess import *
from tqdm import tqdm
import argparse

def preprocess_data(data):
    try:
        ohe_data = ohe.transform(data[ohe_columns].astype(str))
    except:
        import pdb; pdb.set_trace()
    ohe_column_names = [f'ohe{i}' for i in range(len(ohe_data[0]))]
    ohe_data = pd.DataFrame(ohe_data, index = data.index, columns = ohe_column_names)
    data = data.join(ohe_data)
    data['skill_idx'] = np.argmax(data[ohe_column_names].to_numpy(), axis = 1)
    # features = ['correct'] * 20 + ['response_time', 'attempt_count', 'hint_count', 'first_action', 'position'] + ohe_column_names
    features = ['skill_idx', 'correct'] + ohe_column_names + ['opportunity'] #+ ['response_time', 'attempt_count', 'hint_count', 'first_action', 'position']
    if 'opportunity' not in data.columns:
        features[-1] = 'correct'
    seqs = data.groupby(['user_id']).apply(lambda x: x[features].values.tolist())
    length = max(seqs.str.len()) + 1
    seqs = seqs.apply(lambda s: s + (length - min(len(s), length)) * [[-1000] * len(features)])
    return seqs

def construct_batches(raw_data, train = False):
    vc = raw_data['user_id'].value_counts()
    if train:
        user_ids = np.random.permutation(vc.index)
    else:
        user_ids = vc.index

    idx, pos = 0, 0
    while len(user_ids[idx * batch_size: (idx + 1) * batch_size]) > 0 and idx <= len(user_ids) // batch_size:
        user_idx = user_ids[idx * batch_size: (idx + 1) * batch_size]
        if 'order_id' in raw_data.columns:
            filtered_data = raw_data[raw_data['user_id'].isin(user_idx)].sort_values(['user_id', 'order_id'])
        else:
            filtered_data = raw_data[raw_data['user_id'].isin(user_idx)].sort_values('user_id', kind = 'stable')
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        X = torch.tensor(batch[:, :-1, ..., 1:], requires_grad=True, dtype=torch.float32).cuda()
        y = torch.tensor(batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32).cuda()
        i = 0
        if torch.isnan(X).any() or torch.isnan(y).any():
            import pdb; pdb.set_trace()
        while X[:, i * block_size: (i + 1) * block_size].shape[1] > 0:
            if i == 0:
                yield X[:, i * block_size: (i + 1) * block_size], y[:, i * block_size: (i + 1) * block_size], False
            else:
                yield X[:, i * block_size: (i + 1) * block_size], y[:, i * block_size: (i + 1) * block_size], True
            i += 1
        pos += i
        idx += 1
        if train:
            idx = pos

def train_test_split(data, skill_list = None):
    if skill_list is not None:
        data = data[data['skill_id'].isin(skill_list)]
    data = data.set_index(['user_id'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_test = data.loc[test_idx].reset_index()
    return data_train, data_test

def evaluate(model, batches):
    ypred, ytrue = [], []
    global batch_size
    batch_size, old_batch = 32, batch_size
    with torch.no_grad():
        for X, y, cache in tqdm(batches):
            X, y = X.cuda(), y.cuda()
            mask = y[..., -1] != -1000
            if cache:
                corrects, latents, params, loss = model(X, output = y.detach(), return_all = True, prior = latents[-1])
            else:
                corrects, latents, params, loss = model(X, output = y.detach(), return_all = True)
            y = y[..., -1].unsqueeze(-1)[mask]
            ypred.append(corrects[mask].ravel().detach().cpu().numpy())
            ytrue.append(y.ravel().detach().cpu().numpy())
            del X, y
        ypred = np.concatenate(ypred)
        ytrue = np.concatenate(ytrue)
    batch_size = old_batch
    return ypred, ytrue #roc_auc_score(ytrue, ypred)

def bkt_benchmark(train_data, test_data, **model_type):
    model = Model()
    model.fit(data = train_data, **model_type, defaults = {'skill_name': 'skill_id'})
    print(model.params().reset_index().shape)
    preds = model.predict(data = data_test)
    preds = preds.groupby('user_id').apply(lambda x: x.iloc[1:])
    ypred, ytrue = preds['correct_predictions'].to_numpy(), preds['correct'].to_numpy()
    auc = roc_auc_score(ytrue, ypred)
    print(ypred.shape)
    acc = (ytrue == ypred.round()).mean()
    rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
    return auc, acc, rmse

def test_rules(model):
    import pdb; pdb.set_trace()
    params = torch.sigmoid(model.skill_params(model.skill_emb(torch.arange(self.n_skills).cuda())))[..., 1:]
    return (params[:, 3] > 0.5).sum() / num_skills, (params[:, 2] > 0.5).sum() / num_skills, (params[:, 0] > (1 - params[:, 3])/ params[:, 2]).sum() / num_skills

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--block_size",
        type = int,
        default=2048,
    )
    parser.add_argument(
        "--num_layers",
        type = int,
        default=2,
    )
    parser.add_argument(
        "--n_embd",
        type = int,
        default=200,
    )
    parser.add_argument(
        "--data",
        type = str,
        required = True,
    )
    parser.add_argument(
        "--tag",
        type = str,
        required = True,
    )

    args = parser.parse_args()

    fn = args.data
    batch_size, block_size = args.batch_size, args.block_size
    tag = args.tag + '_' + '_'.join(str(datetime.datetime.now()).split()) # sys.argv[1]

    data_train, data_val, data_test = preprocess(pd.read_csv(fn, encoding = 'latin'), impute_template = False)

    #print(bkt_benchmark(data_train, data_test))
    #print(bkt_benchmark(data_train, data_test, forgets = True))
    #print(bkt_benchmark(data_train, data_test, multigs = True))

    print("Train-test split complete...")
    ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')
    ohe_columns = ['skill_id'] #, 'first_action','sequence_id', 'template_id']
    ohe.fit(data_train[ohe_columns].astype(str))
    print("OHE complete...")

    #dataset_train = KTDataset(data_train, ohe)
    #dataset_test = KTDataset(data_test, ohe)
    #batches_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_batch, shuffle=True, num_workers = 4)
    #batches_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate_batch, shuffle=False, num_workers = 4)
    batches_train = construct_batches(data_train)
    batches_test = construct_batches(data_test)


    config = GPTConfig(vocab_size = len(ohe.get_feature_names_out()) * 4, block_size = block_size, n_layer = args.num_layers, n_head = 4, n_embd = args.n_embd, input_size = len(ohe.get_feature_names_out()) + 2, bkt = True)
    model = GPT(config).cuda()
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))
    #model.load_state_dict(torch.load('ckpts/bktransformer_as_2023-12-23_21:49:15.663736-model-99-0.8622401595492162-0.8622401595492162-0.371086448431015.pth'))
    #model.load_state_dict(torch.load('ckpts/bktransformer_alg_2023-12-23_21:48:47.798633-model-98-0.7917582398915805-0.7917582398915805-0.3205636739730835.pth'))
    #model.load_state_dict(torch.load('ckpts/bktransformer_alg_2023-12-26_16:57:22.368396-model-5-0.7183438476055611-0.7183438476055611-0.33736422657966614.pth'))
    #model.load_state_dict(torch.load('ckpts/bktransformer_max_2024-01-02_15:43:40.017064-model-99-2-128-0.8903162035324459-0.8903162035324459-0.3409198224544525.pth'))
    #model.eval()
    #batches_test = construct_batches(data_test)
    #ypred, ytrue = evaluate(model, batches_test)
    #import pdb; pdb.set_trace()

    optimizer = optim.AdamW(model.parameters(), lr = 5e-4)
    def train(num_epochs):
        for epoch in range(num_epochs):
            batches_train = construct_batches(data_train, train = True)
            model.train()
            pbar = tqdm(batches_train)
            losses = []
            for X, y, cache in pbar:
                X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                if cache:
                    _, latents, loss = model(X, output = y.detach(), return_latent = True, prior = latents[-1])
                else:
                    _, latents, loss = model(X, output = y.detach(), return_latent = True)
                if torch.isnan(loss).any():
                    print("NaN Loss - Skipping")
                else:
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                pbar.set_description(f"Training Loss: {np.mean(losses)}")

            if epoch % 10 == 0:
                model.eval()
                batches_val = construct_batches(data_val)
                ypred, ytrue = evaluate(model, batches_val)
                auc = roc_auc_score(ytrue, ypred)
                acc = (ytrue == ypred.round()).mean()
                rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
                print(f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]")
                torch.save(model.state_dict(), f"ckpts/{tag}-model-{epoch}-{args.num_layers}-{args.n_embd}-{auc}-{auc}-{rmse}.pth")
            if epoch == num_epochs - 1:
                model.eval()
                batches_test = construct_batches(data_test)
                ypred, ytrue = evaluate(model, batches_test)
                auc = roc_auc_score(ytrue, ypred)
                acc = (ytrue == ypred.round()).mean()
                rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
                print(len(ytrue))
                print(f"Epoch {epoch}/{num_epochs} - [TEST AUC: {auc}] - [TEST ACC: {acc}] - [TEST RMSE: {rmse}]")
                torch.save(model.state_dict(), f"ckpts/{tag}-model-{epoch}-{args.num_layers}-{args.n_embd}-{auc}-{auc}-{rmse}.pth")
    train(100)
    """
    bkt = []
    for _ in range(5):
        bkt.append(bkt_benchmark(data_train, data_test))
    bkt_mlfgs = []
    for _ in range(5):
        bkt_mlfgs.append(bkt_benchmark(data_train, data_test, multigs = 'opportunity', multilearn = 'opportunity', forgets = True))
    bkt_mlf = []
    for _ in range(5):
        bkt_mlf.append(bkt_benchmark(data_train, data_test, multilearn = 'opportunity', forgets = True))
    bkt_mgs = []
    for _ in range(5):
        bkt_mgs.append(bkt_benchmark(data_train, data_test, multigs = 'opportunity', forgets = True))
    """
