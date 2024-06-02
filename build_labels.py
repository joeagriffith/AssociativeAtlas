import numpy as np
import os

data_path = '../Datasets/MindBigData-Imagenet'
info_path = '../Datasets/WordReport-v1.04.txt'

filenames = os.listdir(data_path)

# read comma separated rows of info
class_ids = {}
i = 0
with open(info_path, 'r') as f:
    for line in f:
        class_ids[int(line.strip().split('n')[-1])] = i
        i += 1
class_ids

def filename_to_class_id(filename):
    # split filename by underscore and chose 4th
    class_id = int(filename.split('_')[3][1:])
    return class_ids[class_id]

class_to_indices = {}
for id in class_ids.values():
    class_to_indices[id] = []

classes = []
for i, filename in enumerate(filenames):
    class_id = filename_to_class_id(filename)
    class_to_indices[class_id].append(i)
    classes.append(class_id)

# save class_to_indices
np.save('class_to_indices.npy', class_to_indices)
np.save('classes.npy', classes)