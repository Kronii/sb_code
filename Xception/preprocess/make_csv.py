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
    PICS_ROOT = "/hdd2/vol1/deepfakeDatabases/Celeb-DF-v2/depth_estimation_AdaBins/"   # The dir of cropped training faces
    train_csv = '../../csv_depth/dfgc_train.csv'  # the train split file
    val_csv = '../../csv_depth/dfgc_val.csv'      # the validation split file
    pic_types = os.listdir(PICS_ROOT)
    pic_sum=0
    for pic_type in pic_types:
        pic_types_list = os.listdir(os.path.join(PICS_ROOT, pic_type))
        for video_frames_dir in pic_types_list:
            pics = os.listdir(os.path.join(os.path.join(PICS_ROOT, pic_type), video_frames_dir))
            train_list, val_list = split(pics, shuffle=True, ratio=0.8)
            if pic_type == 'Celeb-synthesis':
                label = str(1)
            else:
                label = str(0)

            #print("train: " + str(train_list) + "\n", "val: " + str(val_list) + "\n" )

            with open(train_csv, 'a', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                for train_pic in train_list:
                    pic_path = os.path.join(os.path.join(os.path.join(PICS_ROOT, pic_type), video_frames_dir), train_pic)
                    csv_writer.writerow([pic_path, label])

            with open(val_csv, 'a', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                for val_pic in val_list:
                    pic_path = os.path.join(os.path.join(os.path.join(PICS_ROOT, pic_type), video_frames_dir), val_pic)
                    csv_writer.writerow([pic_path, label])

            # with open(val_csv, 'a', encoding='utf-8') as f:
            #     csv_writer = csv.writer(f)
            #     for val_pic in val_list:
            #         pics_path = os.path.join(os.path.join(PICS_ROOT, pic_type), val_pic)
            #         pics_name = os.listdir(pics_path)
            #         for pic_name in pics_name:
            #             if "no_face" not in pic_name:
            #                 pic_path = os.path.join(pics_path, pic_name)
            #                 csv_writer.writerow([pic_path, label])

    print(pic_sum)