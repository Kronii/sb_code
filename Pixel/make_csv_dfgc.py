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
    PICS_ROOT = "/hdd2/vol2/deepfakeDatabases/dfgc/images"   # The dir of cropped training faces
    test_csv = './dfgc_test.csv'  # the train split file
    #val_csv = './celeb_val_2.csv' 
    pic_types = os.listdir(PICS_ROOT)

    with open(test_csv, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])

    count_train = 0


    for pic_type in pic_types:
        pics = os.listdir(os.path.join(PICS_ROOT, pic_type))
        if pic_type == 'real_fulls':
            label = str(0.0)
        else:
            label = str(1.0)

        with open(test_csv, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            for train_pic in pics:
                pic_path = os.path.join(os.path.join(PICS_ROOT, pic_type), train_pic)
                csv_writer.writerow([count_train, pic_path, label])
                count_train += 1
