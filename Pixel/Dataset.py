import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from random import shuffle
import pandas as pd
import cv2 as cv
import gc
from torch.utils import data



class PixWiseDataset(data.Dataset):
    def __init__(self, folders, labels, map_size=14,
                 smoothing=True, transform=None):
        self.transform = transform
        self.map_size = map_size
        self.label_weight = 0.99 if smoothing else 1.0
        self.folders = folders
        self.labels = labels
        #self.data = pd.read_csv(csvfile)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def dataset(self, path, label):
        #for index, ind in enumerate(self.data.index):
        img = Image.open(path)
        #img = cv.resize(np.float32(img), (224, 224))
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = np.moveaxis(img, 2, 0)
        #img = np.asarray(img)

        img = Image.fromarray(np.uint8(img)).convert('RGB')
        if label == 0:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

        if self.transform:
            img = self.transform(img)


        label = np.array(label, dtype=np.float32)

        return [img, mask, label]


    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        try:
            #print(index)
            #print(self.data.iloc[index])
            folder = self.folders[index]
            label = self.labels[index]
        except:
            print('Index is too big: ' + str(index) + ' while max index is: ' + str(len(self.folders)))
        # Load data
        X = self.dataset(folder, label)  # (input) spatial images
        #y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X
