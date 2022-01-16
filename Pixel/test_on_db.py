import cv2 as cv
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
import PIL.Image as Image
import pandas as pd
from Metrics import predict, test_accuracy, test_loss
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, auc
from utils import load_labels
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt

class PixWiseDataset(Dataset):
    def __init__(self, folders, info, map_size=14,
                 smoothing=True, transform=None):
        self.transform = transform
        self.map_size = map_size
        self.label_weight = 0.99 if smoothing else 1.0
        self.folders = folders
        self.info = info
        #self.data = pd.read_csv(csvfile)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_crop_face(self, img_path):
        img = cv.imread(img_path)
        img_name = img_path.split('/')[len(img_path.split('/'))-1]
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        # landms = info[img_name]['landms']
        box = self.info[img_name]['box']
        height, width = img.shape[:2]
        # enlarge the bbox by 1.3 and crop
        scale = 1.1
        # if len(box) == 2:
        #     box = box[0]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        #cv.imwrite('./test/' + img_name + '.png', cropped_face)
        return cropped_face

    def dataset(self, path):
        #for index, ind in enumerate(self.data.index):
        #img = Image.open(path)
        img = self.read_crop_face(path)
        #img = cv.resize(np.float32(img), (224, 224))
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = np.moveaxis(img, 2, 0)
        #img = np.asarray(img)

        img = Image.fromarray(np.uint8(img)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img


    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        try:
            #print(index)
            #print(self.data.iloc[index])
            folder = self.folders[index]
        except:
            print('Index is too big: ' + str(index) + ' while max index is: ' + str(len(self.folders)))
        # Load data
        X = self.dataset(folder)  # (input) spatial images
        #y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X

def calc_prevalence(y_actual):
    return (sum(y_actual) / len(y_actual))


def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS_1.pth'))
model.eval()
model.to('cuda')

tfms = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

train_labels, train_paths = load_labels('./uadfv_test.csv')
#json_file = '/hdd2/vol1/deepfakeDatabases/cropped_videos/Celeb-DF-v2/test/I-frames_meta.json'
#json_file = '/hdd2/vol1/deepfakeDatabases/cropped_videos/UADFV/30-frames_meta.json'
json_file = '/hdd2/vol1/deepfakeDatabases/cropped_videos/UADFV/30-frames_meta.json'

image_paths = list()
labels = list(train_labels)
with open(json_file, 'r') as load_f:
            json_info = json.load(load_f)

dataset_eval = PixWiseDataset(train_paths, json_info, transform=tfms)
dataloader_eval = DataLoader(dataset_eval, batch_size=16,
                                        shuffle=False, num_workers=15)
# USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
# Do set drop_last=False (default) in the above dataloader to ensure all images processed

#print('Detection model inferring ...')
prediction = list()
with torch.no_grad():  # Do USE torch.no_grad()
    for image in tqdm(dataloader_eval):
        image = image.to('cuda:0')
        #mask, binary = model(image)

        masks, binary = model.forward(image)
        for ix, mask in enumerate(masks):    
            res = torch.mean(mask).item()
        #print(str(res) + ' ' + str(binary.item()) + str(' should be fake! (> 0.75)' if labels[ix] else ' should be real (< 0.75)!'))
            prediction.append(res)

prediction = np.array(prediction)
#prediction = torch.cat(prediction, dim=0)
#prediction = prediction.cpu().numpy()
prediction = prediction.squeeze().tolist()
assert isinstance(prediction, list)
assert isinstance(labels, list)
print(len(prediction))
print(len(labels))
assert len(prediction) == len(labels)

fake_counter = 0
for i in labels:
    fake_counter += i

threshold = fake_counter/len(prediction)
print("Threshold is " + str(threshold))

print_report(np.array(labels), np.array(prediction), threshold)

labels=np.array(labels)

score = roc_auc_score(labels, prediction)

import sys

fpr, tpr, _ = roc_curve(labels, prediction)
roc_auc = auc(fpr, tpr)
plt.figure()

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('roc_krivulja_2')

print('AUC score is %f' % score)

