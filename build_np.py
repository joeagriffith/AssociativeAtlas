import csv
import numpy as np
import os
from tqdm import tqdm

path = '../Datasets/MindBigData-Imagenet'
len(os.listdir(path))
data = []
for item in tqdm(os.listdir(path)):
    path_i = os.path.join(path, item)
    csv_data = list(csv.reader(open(path_i, 'r')))
    np_data = np.array(csv_data)[:,1:].astype(float)
    data.append(np_data)
    # close file
    csv_data = None

min_len = 100000
for d in data:
    min_len = min(min_len, d.shape[1])
min_len
# truncate data
for i in range(len(data)):
    data[i] = data[i][:,:min_len]
data = np.array(data)

# save data
np.save('data.npy', data)