from PIL import Image
import cv2
import os
import numpy as np
from torchvision import transforms
from network.transform import xception_transforms_train
from network.transform import xception_transforms_validation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import cm
from torch.utils import data
import torch
from pywt import dwt2

## ---------------------- Dataloaders ---------------------- ##
class Dataset_Csv(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, folders, labels, transform=None, train=True):
        "Initialization"
        # self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.train = train
        
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def dwt2(self, image):
        wavename1 = 'haar'
        wavename2 = 'bior1.3'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to float for more resolution for use with pywt
        image = np.float32(image)
        image /= 255

        cA,(cH, cV, cD) = dwt2(image, wavename1)
        cAA, (cAH, cAV, cAD) = dwt2(cA, wavename1)

        multi_channel_pic = []

        for ix, i in enumerate([cAA, cAH, cAV, cAD, cH, cV, cD]):
            i = cv2.convertScaleAbs(i, alpha=(255.0/i.max()))
            i = cv2.resize(i, (299, 299), interpolation = cv2.INTER_CUBIC)
            multi_channel_pic.append(i)

        multi_channel_pic = np.array(multi_channel_pic)
        multi_channel_pic = multi_channel_pic.flatten()
        multi_channel_pic = multi_channel_pic.reshape(299, 299, 7)

        return multi_channel_pic

    def adaBins(self, img):
        img = img.resize((640,480))
        bin_centers, predicted_depth = infer_helper.predict_pil(img)
        new_image = predicted_depth[0][0]
        print(type(new_image))

        return Image.fromarr(new_image)



    def read_images(self, path, use_transform):
        image = Image.open(path).convert('RGB')
        image = self.adaBins(image)
        print("saved: " + '../slikce/' + path[-6:])
        image.save(('../slikce/' + path[-6:]))
        if use_transform is not None:
            image = use_transform(image)
        return image
        #npimage = cv2.imread(path)
        #image = Image.open(path)
        #npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2RGB)
        #mage = Image.open(path)
        # if(self.train):
        #     npimage = np.asarray(image)
        #     if(random.random() < 0.7)
        #     image = remove_landmark(image, landmarks)

        #     image = Image.fromarr(npimage)
        #     print("saved: " + '../slikce/' + path[-6:])
        #     image.save(('../slikce/' + path[-6:]))

        #print(path)
        #if use_transform is not None:
        #    image = use_transform(image)
            #image = self.dwt2(npimage)
            #image = xception_transforms(npimage)
            #image = np.asarray(image)
            #npimage = use_transform(image=npimage)["image"]
            #try:
            #    image = use_transform(image)
            #except: 
            #    print(path)
            #PILimage = Image.fromarr(image)
            #image = np.asarray(image)
            #image = torch.Tensor(image)
            #image.Normalize([0.5] * 3, [0.5] * 3)
            #print("saved: " + '../slikce/' + path[-6:])
            #PILimage.save(('../slikce/' + path[-6:]))
        #return image

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        try:
            folder = self.folders[index]
        except:
            print('Index is too big: ' + str(index) + ' while max index is: ' + str(len(self.folders)))
        # Load data
        X = self.read_images(folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


## ---------------------- end of Dataloaders ---------------------- ##



