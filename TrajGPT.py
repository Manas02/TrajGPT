#!/usr/bin/env python
# coding: utf-8

# Project : TrajGPT
# Author : Manas Mahale <manas.mahale@bcp.edu.in>

import numpy as np
import torch
from torch.utils.data import Dataset
from model.model import TrajGPT, TrajGPTConfig
from model.trainer import Trainer, TrainerConfig
from model.utils import sample
from model.utils import seed

seed(42)

with np.load('./data/1_alanine-dipeptide-3x250ns-backbone-dihedrals.npz') as data:
    x = data['arr_0'][:, 0]
#     data_1 = data['arr_1']
#     data_2 = data['arr_2']

x = [float(f'{i:.3f}') for i in x]


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = [i/1e3 for i in range(-4000, 4000)]
        vocab_size = len(chars)
        print(f'vocab has {vocab_size} unique characters.')    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1] # grab a chunk of (block_size + 1) characters from the data
        dix = [self.stoi[s] for s in chunk] # encode every character to an integer
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


block_size = 64 # spatial extent of the model for its context

text = x
train_dataset = CharDataset(text, block_size)

mconf = TrajGPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = TrajGPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=10, batch_size=64, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      ckpt_path = None, num_workers=0)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()

context = [x[0]]
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
