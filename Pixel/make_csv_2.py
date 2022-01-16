import os
import csv
import json
import random


def split(full_list, shuffle=True, ratio=0.8):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


if __name__ == '__main__':
    # Modify the following directories to yourselves
    PICS_ROOT = "/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/cropped_images"   # The dir of cropped training faces
    train_csv = './celeb_train_2.csv'  # the train split file
    val_csv = './celeb_val_2.csv' 
    pic_types = os.listdir(PICS_ROOT)

    with open(train_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])


    with open(val_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])

    count_val = 0
    count_train = 0


    for pic_type in pic_types:
        pics = os.listdir(os.path.join(PICS_ROOT, pic_type))
        train_list, val_list = split(pics, shuffle=True, ratio=0.8)
        if pic_type == 'Celeb-synthesis':
            label = str(1.0)
        else:
            label = str(0.0)

        with open(train_csv, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            for train_pic in train_list:
                pics_path = os.path.join(os.path.join(PICS_ROOT, pic_type), train_pic)
                pics_name = os.listdir(pics_path)
                for pic_name in pics_name:
                    pic_path = os.path.join(pics_path, pic_name)
                    csv_writer.writerow([count_train, pic_path, label])
                    count_train += 1

        with open(val_csv, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            for val_pic in val_list:
                pics_path = os.path.join(os.path.join(PICS_ROOT, pic_type), val_pic)
                pics_name = os.listdir(pics_path)
                for pic_name in pics_name:
                    pic_path = os.path.join(pics_path, pic_name)
                    csv_writer.writerow([count_val, pic_path, label])
                    count_val += 1