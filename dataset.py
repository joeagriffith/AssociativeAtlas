import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SpectrogramTripletDataset(Dataset):
    def __init__(self, data, class_to_indices):
        # data: a np array of shape (N, C, H, W)
        # class_to_indices: a dictionary, key: class, values: [idx1, idx2, ...]

        # dataset of spectrograms, (idx, channel, H, W)
        self.data = torch.Tensor(data)

        self.class_to_indices = class_to_indices
        
        self.idx_to_class = {}
        for class_, indices in self.class_to_indices.items():
            for idx in indices:
                self.idx_to_class[idx] = class_
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # get another spectrogram from same class
        class1 = self.idx_to_class[idx]
        same_idx = idx
        possible_idxs = self.class_to_indices[class1]
        i = 0
        while same_idx == idx:
            same_idx = possible_idxs[torch.randint(0, len(possible_idxs), (1,)).item()]
            i += 1
            if i > 100:
                Warning(f"Cannot find two different spectrograms in the same class, class: {class1}")
                break
        
        # get spectrogram from a different class
        # find different class
        class2 = class1
        i = 0
        while class2 == class1:
            class2 = self.classes[torch.randint(0, len(self.classes), (1,)).item()]
            i += 1
            if i > 100:
                Warning(f"Cannot find two different classes, class: {class1}")
                break
        # select random spectrogram from class2
        diff_idx = self.class_to_indices[class2][torch.randint(0, len(self.class_to_indices[class2]), (1,)).item()] 

        return self.data[idx], self.data[same_idx], self.data[diff_idx]