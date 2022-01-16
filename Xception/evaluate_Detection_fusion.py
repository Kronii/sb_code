#!/usr/bin/env python
import model as baseDet
#from submission_Det21 import model as baseDet
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

#img_dir = "/hdd/deepfakeDatabases/dfdc/test/cropped_images"
#json_file = '/hdd/deepfakeDatabases/dfdc/test/test_data_dfdc.json'

#img_dir = "/hdd/deepfakeDatabases/UADFV/images_from_video"
#json_file = '/hdd/deepfakeDatabases/UADFV/UADF_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/UADFV_results.json'

#img_dir = "/hdd/deepfakeDatabases/FaceShifter/images_from_video"
#json_file = '/hdd/deepfakeDatabases/FaceShifter/Face_Shifter_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/FaceShifter_results.json'

#img_dir = "/hdd/deepfakeDatabases/DFDGoogle/images_from_video"
#json_file = '/hdd/deepfakeDatabases/DFDGoogle/DFD_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/DFD_results.json'

img_dir = "/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/depth_estimation_AdaBins_test_cropped"
json_file = '/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/Celeb-DF-v2_META.json'
#json_file_skip = '/home/andrejkronovsek/dfgc/skip-ganomaly-test/Celeb-DF-v2_results.json'



#img_dir = "./../TestDeepTomCruiseBoth/"
#json_file = './DeepTomCruise_test_meta.json'

#img_dir = "./sample_imgs"
#json_file = 'sample_meta.json'

#img_dir = "./../test_data_dfdc/"
#json_file = './test_data_dfdc.json'


# load detection model
#print('loading baseline detection model...')
detModel = baseDet.Model()

#print('Detecting images ...')
img_names, prediction = detModel.run(img_dir, json_file)#, json_file_skip)
assert isinstance(prediction, list)
assert isinstance(img_names, list)

#print(prediction)

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

fpr, tpr, _ = roc_curve(labels, prediction)
roc_auc = auc(fpr, tpr)
plt.figure()

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive rate')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('roc_krivulja')

print('AUC score is %f' % score)