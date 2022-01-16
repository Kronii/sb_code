import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, RandomRotation
import numpy as np
from torch.utils.data import DataLoader
from Dataset import PixWiseDataset
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
from Trainer import Trainer
import gc
from utils import make_weights_for_balanced_classes, load_labels
from torch.utils.data import WeightedRandomSampler

model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS_1.pth'))

loss_fn = PixWiseBCELoss()

print("loss function loaded")

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

print("Garbage collection enabled.\n")
gc.enable()
gc.collect()

print("loading PixWiseDatasets...")

train_csv = './celeb_train_2_smol.csv'
val_csv = './celeb_val_2_smol.csv'

train_labels, val_labels, train_paths, val_paths = load_labels(train_csv, val_csv)

train_dataset = PixWiseDataset(train_paths, train_labels, transform=train_tfms)
#train_dataset = PixWiseDataset('./test_data.csv', transform=train_tfms)
#train_ds = train_dataset.dataset()
#print(len(train_ds))
#labels = [data[2] for data in train_ds]

print("train dataset loaded")

val_dataset = PixWiseDataset(val_paths, val_labels, transform=test_tfms)
#val_dataset = PixWiseDataset('./train_data.csv', transform=test_tfms)
#val_ds = val_dataset.dataset()

print("test dataset loaded")
gc.collect()

batch_size = 16

images_datasets = {}
images_datasets['train'] = train_labels
images_datasets['val'] = val_labels

weights = {x: make_weights_for_balanced_classes(images_datasets[x], stage=x) for
            x in ['train', 'val']}
data_sampler = {x: WeightedRandomSampler(weights[x], len(images_datasets[x]), replacement=True) for x in
                ['train', 'val']}

train_dl = DataLoader(train_dataset, batch_size, sampler=data_sampler['train'], num_workers=15, pin_memory=True)

print("data loader test done")
val_dl = DataLoader(val_dataset, batch_size, sampler=data_sampler['val'], num_workers=15, pin_memory=True)
print("data loader val done")

#sample_weights = make_weights_for_balanced_classes(labels, 2)


# for x, y, z in val_dl:
# 	_, zp = model(x)
# 	print(zp)
# 	print (z)
# 	break

# print(test_accuracy(model, train_dl))
# print(test_loss(model, train_dl, loss_fn))

# 5 epochs ran
epochs = 10
trainer = Trainer(train_dl, val_dl, model, epochs, opt, loss_fn, weights, device='cuda')

print('Training Beginning\n')
trainer.fit()
print('\nTraining Complete')
torch.save(model.state_dict(), './DeePixBiS_2.pth')
