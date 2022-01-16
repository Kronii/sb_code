import os
import os.path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import torchvision.transforms as Transforms
from torch.utils.data import dataset, dataloader
import torch.nn as nn
from network.xception import Xception
import json
from imutils import face_utils
import sys
from pywt import dwt2

class FolderDataset(dataset.Dataset):
    def __init__(self, img_folder, face_info):
        self.img_folder = img_folder
        self.imgNames = sorted(os.listdir(img_folder))
        #self.imgNames = [item for item in self.imgNames for i in range(68)]
        # REMEMBER to use sorted() to ensure correct match between imgNames and predictions
        # do NOT change the above two lines

        self.face_info = face_info

        self.transform_picture = Transforms.Compose([
            Transforms.Resize((299, 299)),
            Transforms.ToTensor(),
            Transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.imgNames)

    def adaBins(self, img):
        img = img.resize((640,480))
        print("starting to shit")
        bin_centers, predicted_depth = infer_helper.predict_pil(img)
        print("ended shitting")
        new_image = predicted_depth[0][0]
        print(type(new_image))

        return Image.fromarr(new_image)

    def read_crop_face(self, img_name, img_folder, info):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        # landms = info[img_name]['landms']
        box = info[img_name]['box']
        height, width = img.shape[:2]
        # enlarge the bbox by 1.3 and crop
        scale = 1.3
        # if len(box) == 2:
        #     box = box[0]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        return cropped_face

    def dwt2(self, image):
        wavename1 = 'haar'
        wavename2 = 'bior1.3'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to float for more resolution for use with pywt
        image = np.float32(image)
        image /= 255

        cA,(cH, cV, cD) = dwt2(image, wavename1)
        cAA, (cAH, cAV, cAD) = dwt2(cA, wavename1)

        multi_channel_pic = []

        for ix, i in enumerate([cAA, cAH, cAV, cAD, cH, cV, cD]):
            i = cv2.convertScaleAbs(i, alpha=(255.0/i.max()))
            i = cv2.resize(i, (299, 299), interpolation = cv2.INTER_CUBIC)
            multi_channel_pic.append(i)

        multi_channel_pic = np.array(multi_channel_pic)
        multi_channel_pic = multi_channel_pic.flatten()
        multi_channel_pic = multi_channel_pic.reshape(299, 299, 7)

        return multi_channel_pic

    def __getitem__(self, idx):
        img_name = self.imgNames[idx]
        # Read-in images are full frames, maybe you need a face cropping.
        
        #img = self.read_crop_face(img_name, self.img_folder, self.face_info)
        #img = Image.fromarray(img)
        img = cv2.imread(os.path.join(self.img_folder, img_name))
        #img = self.dwt2(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform_picture(img)

        return img


class Model_Xception():
    def __init__(self, modelName):
        # init and load your model here
        model = Xception()
        model.fc = nn.Linear(2048, 1)
        thisDir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
        model.load_state_dict(torch.load(os.path.join(thisDir, modelName)))
        model.eval()
        model.cuda()
        self.model = model

class Model():
    def __init__(self):
        xception = Model_Xception('/hdd2/vol2/models/training_depth/3_xception.ckpt')
        self.model_xception = xception.model
        self.batchsize = 64


    def run(self, input_dir, json_file):
        
        with open(json_file, 'r') as load_f:
            json_info = json.load(load_f)
        dataset_eval = FolderDataset(input_dir, json_info)
        dataloader_eval = dataloader.DataLoader(dataset_eval, batch_size=self.batchsize,
                                                shuffle=False, num_workers=1)
        # USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
        # Do set drop_last=False (default) in the above dataloader to ensure all images processed

        #print('Detection model inferring ...')
        prediction = []
        with torch.no_grad():  # Do USE torch.no_grad()
            for image in tqdm(dataloader_eval):
                image = image.to('cuda:0')
                outputs_xception = self.model_xception(image)
                preds_xception = torch.sigmoid(outputs_xception)

                prediction.append(preds_xception)


        prediction = torch.cat(prediction, dim=0)
        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().tolist()
        assert isinstance(prediction, list)
        assert isinstance(dataset_eval.imgNames, list)
        assert len(prediction) == len(dataset_eval.imgNames)

        return dataset_eval.imgNames, prediction
