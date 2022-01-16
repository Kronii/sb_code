import torch.nn as nn
from torch import Tensor, from_numpy
from Metrics import test_accuracy, test_loss


class Trainer():
    def __init__(self, train_dl, val_dl, model, epochs, opt, loss_fn, sample_weights, device='cpu'):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model.to(device)
        self.epochs = epochs
        self.opt = opt
        self.loss_fn = loss_fn
        self.device = device
        self.weights = sample_weights

    def train_one_epoch(self, num):
        print(f'\nEpoch ({num+1}/{self.epochs})')
        print('----------------------------------')
        #self.model.train()
        for ix, (img, mask, label) in enumerate(self.train_dl):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.model(img)
            loss = self.loss_fn(net_mask, net_label, mask, label)
            #s_weights = from_numpy(self.weights['train']).float().to(self.device)
            s_weights = self.weights['train'].to(self.device)
            weighted_loss = (loss*s_weights).mean()

            # Train
            self.opt.zero_grad()
            weighted_loss.backward()
            self.opt.step()

            if ix % 9 == 0:
                print(f'Loss : {weighted_loss} at index : {ix} of {len(self.train_dl)}')

        # self.model.eval()
        test_acc = test_accuracy(self.model, self.val_dl)
        test_los = test_loss(self.model, self.val_dl, self.loss_fn)

        print(f'Test Accuracy : {test_acc}  Test Loss : {test_los}')
        return test_acc, test_los

    def fit(self):
        training_loss = []
        training_acc = []
        self.model.train()
        for epoch in range(self.epochs):
            train_acc, train_loss = self.train_one_epoch(epoch)
            training_acc.append(train_acc)
            training_loss.append(train_loss)

        return training_acc, training_loss
