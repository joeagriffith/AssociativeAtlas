import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class SpectrogramTripletDataset(Dataset):
    def __init__(self, data, labels):
        # data: a tensor of shape (N, C, H, W)
        # classes: a tensor of shape (N,)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # get another spectrogram from same class
        class1 = self.labels[idx]
        possible_idxs = torch.where(self.labels == class1)[0]
        same_idx = possible_idxs[torch.randint(0, len(possible_idxs), (1,)).item()].item()
        i = 0
        while same_idx == idx:
            same_idx = possible_idxs[torch.randint(0, len(possible_idxs), (1,)).item()]
            i += 1
            if i > 100:
                Warning(f"Cannot find two different spectrograms in the same class, class: {class1}")
                break
        
        # get spectrogram from a different class
        # find different class
        class2 = self.labels[torch.randint(0, len(self.labels), (1,)).item()]
        i = 0
        while class2 == class1:
            class2 = self.labels[torch.randint(0, len(self.labels), (1,)).item()]
            i += 1
            if i > 100:
                Warning(f"Cannot find two different classes, class: {class1}")
                break
        # select random spectrogram from class2
        possible_idxs = torch.where(self.labels == class2)[0]
        diff_idx = possible_idxs[torch.randint(0, len(possible_idxs), (1,)).item()]

        return self.data[idx], self.data[same_idx], self.data[diff_idx]
    
    def __len__(self):
        return len(self.labels)
    
    def give_val_set(self, split_ratio=0.2):
        # shuffle
        idxs = torch.randperm(len(self.data))
        self.data = self.data[idxs]
        self.labels = self.labels[idxs]

        split_idx = int(len(self.data) * split_ratio)
        self.data = self.data[split_idx:]
        self.labels = self.labels[split_idx:]
        return SpectrogramTripletDataset(self.data[:split_idx], self.labels[:split_idx])