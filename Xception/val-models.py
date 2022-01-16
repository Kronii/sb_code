
import os
import time
import stat
import csv
import sys
import numpy as np


from network.xception import xception
from network.transform import xception_transforms_train
from network.transform import xception_transforms_validation
import torch.nn as nn
from torch.utils import data
import torch
from network.utils import Dataset_Csv

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix

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

#  Input path of models
path = '../training_x68_no_aug_v5_100/fake_detect/xception'
val_csv = "../../csv/dfgc_val_x68.csv"        # The validation split file

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
        running_loss_val += batch_loss * len(labels_val)

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

    epoch_loss = running_loss_val / len(test_list)
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


files_sorted_by_date = []

filepaths = [os.path.join(path, file) for file in os.listdir(path) if os.path.splitext(file)[1]=='.ckpt']

file_statuses = [(os.stat(filepath), filepath) for filepath in filepaths]

files = ((status[stat.ST_CTIME], filepath) for status, filepath in file_statuses if stat.S_ISREG(status[stat.ST_MODE]))


#log
log = Logger('output.log', sys.stdout)

batch_size = 164
sfiles=sorted(files)

for creation_time, filepath in sfiles:
    if filepath.endswith('.log'):
        continue
    creation_date = time.ctime(creation_time)
    filename = os.path.basename(filepath)
    files_sorted_by_date.append(creation_date + " " + filename + "\n")
    log.write(filename+"\n")

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}




    model = xception()
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join(filepath)))
    model.eval()
    model.cuda(0)

    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()

    test_list = []
    test_label = []
    validation_data(val_csv)


    valid_set = Dataset_Csv(test_list, test_label, transform=xception_transforms_validation)
    images_datasets_test = test_label
    image_datasets_test = data.DataLoader(valid_set, batch_size, **params)
    
    dataloaders = {'test': image_datasets_test}
    datasets_sizes = {'test': len(image_datasets_test)}


    test_loss = val_models(model, criterion, 5, test_list, 0)
    
    log.write("test_loss = "+str(test_loss)+"\n")
    






