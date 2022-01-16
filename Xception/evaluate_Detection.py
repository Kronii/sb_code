#!/usr/bin/env python
from submission_Det import model as baseDet
from sklearn.metrics import roc_auc_score

#img_dir = "/hdd/deepfakeDatabases/UADFV/images_from_video"
#json_file = '/hdd/deepfakeDatabases/UADFV/UADF_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/UADFV_results.json'

#img_dir = "/hdd/deepfakeDatabases/FaceShifter/images_from_video"
#json_file = '/hdd/deepfakeDatabases/FaceShifter/Face_Shifter_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/FaceShifter_results.json'

#img_dir = "/hdd/deepfakeDatabases/DFDGoogle/images_from_video"
#json_file = '/hdd/deepfakeDatabases/DFDGoogle/DFD_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/DFD_results.json'

img_dir = "/hdd/deepfakeDatabases/Celeb-DF-v2/images_from_video_test"
json_file = '/hdd/deepfakeDatabases/Celeb-DF-v2/Celeb-DF-v2_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/Celeb-DF-v2_results.json'


# load detection model
print('loading baseline detection model...')
detModel = baseDet.Model()

print('Detecting images ...')
img_names, prediction = detModel.run(img_dir, json_file)

assert isinstance(prediction, list)
assert isinstance(img_names, list)
positive = 0
negative = 0

for name in img_names:
    if name[0] == "1":
        positive = positive + 1
    else:
        negative = negative + 1    

#print(str(positive) + '   ' + str(negative))

labels = [0]*(negative) + [1]*(positive)
score = roc_auc_score(labels, prediction)

import sys

print('AUC score is %f' % score)
