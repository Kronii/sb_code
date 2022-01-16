import numpy as np
import csv
import torch


def make_weights_for_balanced_classes(labels, stage='train'):
    # count = np.zeros(nclasses)
    # for item in labels:
    #     count[int(item)] += 1
    # print(count)
    # weight_per_class = np.zeros(nclasses)
    # N = float(sum(count))
    # for i in range(nclasses):
    #     weight_per_class[i] = N / float(count[i])
    # weight = np.zeros(len(labels))
    # for idx, val in enumerate(labels):
    #     weight[idx] = weight_per_class[int(val)]
    # return weight

    targets = []

    targets = torch.tensor(labels)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


def load_labels(train_csv, val_csv=None):
    frame_reader = open(train_csv, 'r')
    csv_reader = csv.reader(frame_reader)
    train_labels = list()
    train_paths = list()

    for ix, f in enumerate(csv_reader):
        if ix==0:
            continue
        train_paths.append(f[1])
        label = int(float(f[2]))
        train_labels.append(label)

    frame_reader.close()

    if val_csv != None:
        frame_reader = open(val_csv, 'r')
        csv_reader = csv.reader(frame_reader)
        val_labels = list()
        val_paths = list()

        for ix, f in enumerate(csv_reader):
            if ix==0:
                continue
            val_paths.append(f[1])
            label = int(float(f[2]))
            val_labels.append(label)

        return train_labels, val_labels, train_paths, val_paths
    else:
        return train_labels, train_paths