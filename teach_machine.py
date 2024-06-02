import torch
from data.dataset import SpectrogramTripletDataset
from model import CNN, Transformer
import numpy as np
from optim.functional import get_optimiser
from optim.train import train_contrastive, train_classify

config = {
    'lr': 3e-4,
    'num_epochs': 100,
    'batch_size': 64,
    'num_features': 64,
    'embed_dim': 64,
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

model_name = f'cnn_classify-{config["num_features"]}-{config["embed_dim"]}'
model = CNN(config['num_features'], config['embed_dim']).to(device)
# model = Transformer(config['num_features']*2, config['num_features']).to(device)
optimiser = get_optimiser(model, 'AdamW', config['lr'], 0.04)


# train_contrastive(
#     model,
#     optimiser,
#     train_dataset,
#     val_dataset,
#     config['batch_size'],
#     config['num_epochs'],
#     config,
#     log_dir=f'out/logs/{model_name}/')

train_classify(
    model,
    optimiser,
    train_dataset,
    val_dataset,
    config['batch_size'],
    config['num_epochs'],
    config,
    log_dir=f'out/logs/{model_name}/',
    save_dir=f'out/models/{model_name}.pth'
)

# save model
torch.save(model.state_dict(), 'out/models/{model_name}.pth')