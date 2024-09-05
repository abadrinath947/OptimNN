"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.3

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # self.model_type = config.model_type

        # input embedding stem
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head

        self.n_skills = config.vocab_size // 4
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), nn.PReLU(), nn.Linear(config.n_embd, config.n_embd),
                                  nn.PReLU(), nn.Linear(config.n_embd, config.vocab_size))

        self.block_size = config.block_size
        # self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        #self.obs_emb = nn.Sequential(nn.Linear(125, config.n_embd), nn.Tanh())
        #self.obs_emb = nn.Sequential(nn.Linear(1605, config.n_embd), nn.Tanh())
        self.obs_emb = nn.Sequential(nn.Linear(config.input_size, config.n_embd), nn.PReLU(), nn.Linear(config.n_embd, config.n_embd),
                                     nn.PReLU(), nn.Linear(config.n_embd, config.n_embd))

        if self.config.bkt:
            layers = []
            for _ in range(4):
                layers.extend([nn.Linear(config.n_embd, config.n_embd), nn.ReLU()])
            layers.extend([nn.Linear(config.n_embd, 5)])
            self.skill_params = nn.Sequential(*layers).cuda()
            self.skill_emb = nn.Embedding(self.n_skills, config.n_embd)
            self.correct_emb = nn.Embedding(2, config.n_embd)
        else:
            self.post = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.PReLU(), nn.Linear(config.n_embd, config.n_embd),
                                     nn.PReLU(), nn.Linear(config.n_embd, config.vocab_size), nn.PReLU())

    def get_block_size(self):
        return self.block_size

    # state, action, and return
    def forward(self, obs, return_latent = False, return_all = False, output = None, params = None, prior = None, lambd = [50, 50, 50, 1]):
        B, T, D = obs.shape
        if D == 2:
            obs = self.skill_emb(obs[..., 0]) + self.correct_emb(obs[..., 1])
        token_embeddings = self.obs_emb(obs) # (batch * block_size, n_embd)
        # all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logit_diff = self.head(x)
        loss = 0

        if self.config.bkt:
            logits = self.skill_params(self.skill_emb(torch.arange(self.n_skills, device = obs.device)))
            oparams = logits[..., 1:].reshape(1, -1)
            params = torch.sigmoid(oparams + logit_diff).view(B, T, -1, 4)
            oparams = torch.sigmoid(oparams.view(-1, 4))
            loss = loss + lambd[0] * (F.relu(params[..., 0] - (1 - params[..., 3]) / (params[..., 2] + 1e-6)).mean() + F.relu(oparams[..., 0] - (1 - oparams[..., 3]) / (oparams[..., 2] + 1e-6)).mean()) \
                    + lambd[1] * (F.relu(params[..., 2] - 0.5).mean() + F.relu(oparams[..., 2] - 0.5).mean())\
                    + lambd[2] * (F.relu(params[..., 3] - 0.5).mean() + F.relu(oparams[..., 3] - 0.5).mean()) \
                    + lambd[3] * torch.mean(logit_diff ** 2) \
                    + lambd[3] * (torch.mean((logit_diff[:, 1:] - logit_diff[:, :-1]) ** 2) if logit_diff.shape[1] > 1 else 0)

            corrects, latent = torch.zeros_like(params[..., -1]), torch.sigmoid(logits[..., 0].repeat((B, 1)))
            if prior is not None:
                latent = prior.detach()
            latents = []
            for i in range(T):
                latent = torch.clamp(latent, min = 1e-5, max = 1 - 1e-5)
                latents.append(latent)
                correct, latent = self.extract_latent_correct(params[:, i].view(B, -1, 4), latent, 
                        true_correct = output[:, i, -1], 
                        skills = torch.where(output[:, i, 0] == -1000, 0, output[:, i, 0]).long())
                corrects[:, i] = correct.squeeze()
        else:
            corrects = torch.sigmoid(self.post(logit_diff.view(B, T, -1)))
            latents = [None]
            params = [None]

        if output is not None:
            skill_idx = torch.where(output[..., 0] == -1000, 0, output[..., 0])
            if skill_idx.max() >= corrects.shape[-1]:
                import pdb; pdb.set_trace()
            corrects = torch.gather(corrects, dim = -1, index = skill_idx.long().unsqueeze(-1))
            mask = output[..., -1] != -1000
            loss = loss + F.binary_cross_entropy(corrects[mask], output[..., -1:][mask])

        if return_latent:
            return corrects, latents, loss
        elif return_all:
            return corrects, latents, params, loss
        return corrects, loss

    def extract_latent_correct(self, params, latent, true_correct, skills):
        l, f, g, s = params[..., 0], 0, params[..., 2], params[..., 3]
        # g, s = 0, 0
        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        k_t = torch.clone(latent)
        k_t[range(len(k_t)), skills] = torch.where(true_correct > 0.5, k_t1[range(len(k_t)), skills], k_t0[range(len(k_t)), skills])
        k_t[range(len(k_t)), skills] = k_t[range(len(k_t)), skills] + (1 - k_t[range(len(k_t)), skills]) * l[range(len(k_t)), skills]
        return correct, torch.clamp(k_t, 1e-4, 1 - 1e-4)

