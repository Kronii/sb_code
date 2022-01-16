from __future__ import print_function, division
import torch.optim as optim
import sys
import time
import numpy as np
import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.utils import data
from network.xception import xception
from network.transform import xception_transforms_train
from network.transform import xception_transforms_validation
from sklearn.metrics import accuracy_score, roc_curve, auc
from random import shuffle
from network.utils import Dataset_Csv
import csv
from sklearn.metrics import confusion_matrix
import cv2
import gc

from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def make_weights_for_balanced_classes(train_dataset, stage='train'):
    targets = []
    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    targets_x68 = targets[::68]
    del targets
    gc.collect()
    samples_weight = torch.tensor([weight[t] for t in targets_x68])
    samples_weight = torch.repeat_interleave(samples_weight, 68)
    #to je treba zaokrožit navzdol pa popravit, ker tut pri cropanju se kaka slika ne shrani od landmarkov in jih ni vedno 68

    log.write('Samples weight size: {}\n'.format(samples_weight.size()))
    
    return samples_weight


def print_mem():
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    log.write('Total RAM: {}, used RAM: {}\n'.format(tot_m, used_m))

def train_model(model, model_dir, criterion, optimizer, scheduler, num_epochs=10, current_epoch=0):
    best_logloss = 10.0
    best_epoch = 0
    for epoch in range(current_epoch, num_epochs):
        best_test_logloss = 10.0
        epoch_start = time.time()
        model_out_path = os.path.join(model_dir, str(epoch) + '_xception.ckpt')
        log.write('------------------------------------------------------------------------\n')
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            print_mem()
            if phase == 'train':
                #scheduler.step() #zakomentiral BB: Warning: you should call `optimizer.step()` before `lr_scheduler.step()`.
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_loss_train = 0.0
            running_loss100 = 0.0

            y_scores, y_trues = [], []
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.cuda(), labels.to(torch.float32).cuda()

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        # labels = labels.unsqueeze(1)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                batch_loss = loss.data.item()
                running_loss += batch_loss
                running_loss_train += batch_loss
                running_loss100 += batch_loss

                y_true = labels.data.cpu().numpy()
                y_score = preds.data.cpu().numpy()

                if i % 100 == 0:
                    writer.add_scalars(phase+'ing loss',{'train':running_loss100 / 100}, epoch * len(dataloaders[phase]) + i)
                    batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
                    running_loss100 = 0.0
                    log.write(
                        'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch,
                                                                                                      num_epochs - 1,
                                                                                                      i, len(
                                dataloaders[phase]), phase, batch_loss, batch_acc))
                if (i + 1) % 3000 == 0 and phase == 'train':
                    print_mem()
                    inter_loss = running_loss_train / 3000.0
                    writer.add_scalars(phase+'ing loss',{'train1':inter_loss}, epoch * len(dataloaders[phase]) + i)
                    log.write('last phase train loss is {}\n'.format(inter_loss))
                    running_loss_train = 0.0
                    #test_loss = val_models(model, criterion, num_epochs, test_list, epoch)
                    #writer.add_scalars(phase+'ing loss',{'val':test_loss}, epoch * len(dataloaders[phase]) + i)
                    # if test_loss < best_test_logloss:
                    #     best_test_logloss = test_loss
                    #     log.write('save current model {}, Now time is {}, best logloss is {}\n'.format(i,time.asctime( time.localtime(time.time()) ),best_test_logloss))
                    #     model_out_paths = os.path.join(model_dir, str(epoch) + str(i) + '_xception.ckpt')
                    #     torch.save(model.module.state_dict(), model_out_paths)
                    #model.train()
                    log.write('save current model {}, Now time is {}\n'.format(i,time.asctime( time.localtime(time.time()) )))
                    model_out_paths = os.path.join(model_dir, str(epoch) + '_' + str(i) + '_xception.ckpt')
                    torch.save(model.module.state_dict(), model_out_paths)
                    # scheduler.step()
                    #log.write('now lr is : {}\n'.format(scheduler.get_lr()))

                    log.write('now lr is : {}\n'.format(scheduler.get_last_lr()))

                if phase == 'test':
                    y_scores.extend(y_score)
                    y_trues.extend(y_true)
            if phase == 'train': #dodal BB
                scheduler.step()
            if phase == 'test':
                epoch_loss = running_loss / (len(test_list) / batch_size)
                writer.add_scalars('training loss',{'val_epoch':epoch_loss}, (epoch+1) * len(dataloaders['train']))
                y_trues, y_scores = np.array(y_trues), np.array(y_scores)
                accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))

                log.write(
                    '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, num_epochs - 1, phase,
                                                                                        epoch_loss,
                                                                                        accuracy))
            if phase == 'test' and epoch_loss < best_logloss:
                best_logloss = epoch_loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), model_out_path)

        log.write('Epoch {}/{} Time {}s\n'.format(epoch, num_epochs - 1, time.time() - epoch_start))
    log.write('***************************************************')
    log.write('Best logloss {:.4f} and Best Epoch is {}\n'.format(best_logloss, best_epoch))

def val_models(model, criterion, num_epochs, test_list, current_epoch=0 ,phase='test'):
    log.write('------------------------------------------------------------------------\n')
    # Each epoch has a training and validation phase
    model.eval()
    running_loss_val = 0.0
    # print(phase)
    y_scores, y_trues = [], []
    for k, (inputs_val, labels_val) in enumerate(dataloaders[phase]):
        inputs_val, labels_val = inputs_val.cuda(), labels_val.to(torch.float32).cuda()
        with torch.no_grad():
            outputs_val = model(inputs_val)
            # labels = labels.unsqueeze(1)
            loss = criterion(outputs_val, labels_val)
            preds = torch.sigmoid(outputs_val)
        batch_loss = loss.data.item()
        running_loss_val += batch_loss

        y_true = labels_val.data.cpu().numpy()
        y_score = preds.data.cpu().numpy()

        if k % 100 == 0:
            batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
            log.write(
                'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch,
                                                                                              num_epochs - 1,
                                                                                              k, len(dataloaders[phase]),
                                                                                              phase, batch_loss, batch_acc))
        y_scores.extend(y_score)
        y_trues.extend(y_true)

    epoch_loss = running_loss_val / (len(test_list) / batch_size)
    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
    accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))
    # model_out_paths = os.path.join(model_dir, str(current_epoch) + '_xception.ckpt')
    # torch.save(model.module.state_dict(), model_out_paths)
    log.write(
        '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch, num_epochs - 1, phase,
                                                                            epoch_loss,
                                                                            accuracy))
    tn, fp, fn, tp = confusion_matrix(y_trues, np.where(y_scores > 0.5, 1, 0)).ravel()
    log.write(
        '**Epoch {}/{} Stage: {} TNR: {:.2f} FPR: {:.2f} FNR: {:.2f} TPR: {:.2f} \n'.format(current_epoch, num_epochs - 1, phase,
                                                                            tn/(fp + tn),fp/(fp + tn),fn/(tp + fn),tp/(tp + fn)))
    log.write('***************************************************\n')
    # model.train()
    return epoch_loss

def base_data(csv_file):
    frame_reader = open(csv_file, 'r')
    csv_reader = csv.reader(frame_reader)

    for f in csv_reader:
        path = f[0]
        label = int(f[1])
        train_label.append(label)
        train_list.append(path)
    log.write(str(len(train_list)) + '\n')

def validation_data(csv_file):
    frame_reader = open(csv_file, 'r')
    fnames = csv.reader(frame_reader)
    for f in fnames:
        path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(path)
    frame_reader.close()
    log.write(str(len(test_label)) + '\n')


if __name__ == '__main__':
    # Modify the following directories to yourselves
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    current_epoch = 0
    batch_size = 100
    train_csv = "../../csv/dfgc_train_x68.csv"    # The train split file
    val_csv = "../../csv/dfgc_val_x68.csv"        # The validation split file

    #  Output path
    model_dir = '../training_x68_less_aug_100'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_name = model_dir.split('/')[-2] + '.log'
    log_dir = os.path.join(model_dir, log_name)
    if os.path.exists(log_dir):
        os.remove(log_dir)
        print('The log file is exit!')

    log = Logger(log_dir, sys.stdout)
    log.write('model : xception   batch_size : 100 frames : 10 \n')
    log.write('pretrain : True   input_size : 48*48\n')

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_list = []
    train_label = []
    log.write('loading train data' + '\n')
    base_data(train_csv)

    ziplist = list(zip(train_list, train_label))
    shuffle(ziplist)
    train_list[:], train_label[:] = zip(*ziplist)

    test_list = []
    test_label = []

    log.write('loading val data' + '\n')
    validation_data(val_csv)

    log.write("Garbage collection enabled.\n")
    gc.enable()

    train_set, valid_set = Dataset_Csv(train_list, train_label, transform=xception_transforms_train), \
                           Dataset_Csv(test_list, test_label, transform=xception_transforms_validation, train=False)

    images_datasets = {}
    images_datasets['train'] = train_label
    images_datasets['test'] = test_label

    weights = {x: make_weights_for_balanced_classes(images_datasets[x], stage=x) for
               x in ['train', 'test']}
    data_sampler = {x: WeightedRandomSampler(weights[x], len(images_datasets[x]), replacement=True) for x in
                    ['train', 'test']}

    image_datasets = {}
    # over sampling
    image_datasets['train'] = data.DataLoader(train_set, sampler=data_sampler['train'], batch_size=batch_size, **params)

    # image_datasets['train'] = data.DataLoader(train_set, batch_size=batch_size, **params)
    image_datasets['test'] = data.DataLoader(valid_set, batch_size=batch_size, **params)

    dataloaders = {x: image_datasets[x] for x in ['train', 'test']}
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    gc.collect()

    model = xception(pretrained=True)
    model.train()
    model = nn.DataParallel(model.cuda())
    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.00035, weight_decay=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.5)

    writer = SummaryWriter('runs/xception_experiment_6')

    train_model(model=model, model_dir=model_dir, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=10,
                current_epoch=current_epoch)

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))
    writer.close()
