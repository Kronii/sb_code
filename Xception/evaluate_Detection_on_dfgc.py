#!/usr/bin/env python
#from submission_xception_7layers import model as baseDet
#from submission_Fusion import model as baseDet

# from modeli_oddani_na_tekmovanje.efficientXception_v2_skip import model as baseDet
import model as baseDet


#from submission_Det21 import model as baseDet
from sklearn.metrics import roc_auc_score
import sys
import os


databases_dir = "/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/images_from_video_test/"
json_file = '/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/Celeb-DF-v2-test.json'
#json_file_pre = '/hdd/deepfakeDatabases/dfgc/images_for_skip_ganomaly/'


# load detection model
#print('loading baseline detection model...')
detModel = baseDet.Model()

sum_all = 0

databases = os.listdir(databases_dir)
labels_all = list()
prediction_all = list()

for database in databases:
    if database[-5:] == '.json':
        continue
    print(database)
    img_dir = os.path.join(databases_dir, database)
    #print(os.path.join(json_file_pre, database + '.json'))
    img_names, prediction = detModel.run(img_dir, json_file)#, os.path.join(json_file_pre, database + '_results.json'))
    assert isinstance(prediction, list)
    assert isinstance(img_names, list)

    #print(prediction)

    positive = 0
    negative = 0

    for name in img_names:
        if "real" not in name:
            positive = positive + 1
        else:
            negative = negative + 1    

    #print(str(positive) + '   ' + str(negative))

    labels = [0]*(negative) + [1]*(positive)
    labels_all.append(labels)
    prediction_all.append(prediction)

prediction_all = [prediction for sublist in prediction_all for prediction in sublist]
labels_all = [label for sublist in labels_all for label in sublist]

score = roc_auc_score(labels_all, prediction_all)

print('AUC score vsekupi je %f' % score)

#print('AUC score vsekupi je %f' % sum_all)