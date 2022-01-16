import os
import csv
import json
import random
import itertools

def split(full_list, shuffle=True, ratio=0.8, dlib_landmarks=0):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if dlib_landmarks > 0:
        offset = offset - (offset%dlib_landmarks)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle and dlib_landmarks == 0:
        random.shuffle(full_list)
    if shuffle and dlib_landmarks > 0:
        list_x68 = [full_list[i * dlib_landmarks:(i + 1) * dlib_landmarks] for i in range((len(full_list) + dlib_landmarks - 1) // dlib_landmarks )]
        random.shuffle(list_x68)
        full_list = list(itertools.chain(*list_x68))


    if dlib_landmarks > 0: 
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:(n_total - (n_total%dlib_landmarks))]

        return sublist_1, sublist_2
    
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:n_total]

    return sublist_1, sublist_2

if __name__ == '__main__':
    # Modify the following directories to yourselves
    PICS_TRAIN_ROOT = "/hdd2/vol1/deepfakeDatabases/cropped_videos/Celeb-DF-v2/test/I-frames"   # The dir of cropped training faces
    PICS_TEST_ROOT = "/hdd2/vol1/deepfakeDatabases/cropped_videos/Celeb-DF-v2/test/I-frames"
    train_csv = './celeb_train.csv'  # the train split file
    val_csv = './celeb_val.csv' 
    test_csv = './celeb_test.csv'      # the test  file
    pics_train = os.listdir(PICS_TRAIN_ROOT)
    pics_test = os.listdir(PICS_TEST_ROOT)
    pic_sum=0

    train_list, val_list = split(pics_train, shuffle=True, ratio=0.8)

    with open(train_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])
        for ix, pic_name in enumerate(train_list):
            if pic_name[0]=='1':
                label = str(1.0)
            else:
                label = str(0.0)
            pic_path = os.path.join(PICS_TRAIN_ROOT, pic_name)
            csv_writer.writerow([ix, pic_path, label])

    with open(val_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])
        for ix, pic_name in enumerate(val_list):
            if pic_name[0]=='1':
                label = str(1.0)
            else:
                label = str(0.0)
            pic_path = os.path.join(PICS_TRAIN_ROOT, pic_name)
            csv_writer.writerow([ix, pic_path, label])

    with open(test_csv, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "name", "label"])
        for ix, pic_name in enumerate(pics_test):
            if pic_name[0]=='1':
                label = str(1.0)
            else:
                label = str(0.0)
            pic_path = os.path.join(PICS_TEST_ROOT, pic_name)
            csv_writer.writerow([ix, pic_path, label])

    print(pic_sum)