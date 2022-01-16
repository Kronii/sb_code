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
    PICS_ROOT = "/hdd2/vol1/deepfakeDatabases/cropped_videos/UADFV/30-frames"   # The dir of cropped training faces
    test_csv = './uadfv_test.csv'  # the train split file
    #val_csv = './celeb_val_2.csv' 
    pics = os.listdir(PICS_ROOT)

    with open(test_csv, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])

    count_train = 0

    with open(test_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for pic_name in pics:
            pic_path = PICS_ROOT + '/' +pic_name
            if pic_name[0] == '1':
                label = str(1.0)
            else:
                label = str(0.0)
            csv_writer.writerow([count_train, pic_path, label])
            count_train += 1
                    
