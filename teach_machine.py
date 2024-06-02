import torch
from data.dataset import SpectrogramTripletDataset
from model import CNN, Transformer
import numpy as np
from optim.functional import get_optimiser
from optim.train import train_contrastive

config = {
    'lr': 0.0001,
    'num_epochs': 100,
    'batch_size': 64,
    'num_features': 64,
    'margin': 0.5,
    'split_ratio': 0.2,
}

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True 

data = np.load('data/spectrograms.npy', allow_pickle=True)
data = torch.tensor(data, dtype=torch.float32).to(device)
labels = np.load('data/classes.npy', allow_pickle=True)
labels = torch.tensor(labels, dtype=torch.long).to(device)
train_dataset = SpectrogramTripletDataset(data, labels)
val_dataset = train_dataset.give_val_set(config['split_ratio'])

# model = CNN(config['num_features']).to(device)
model = Transformer(config['num_features']*2, config['num_features']).to(device)
optimiser = get_optimiser(model, 'AdamW', config['lr'], 0.04)


train_contrastive(
    model,
    optimiser,
    train_dataset,
    val_dataset,
    config['batch_size'],
    config['num_epochs'],
    config,
    log_dir='out/logs/transformer/')