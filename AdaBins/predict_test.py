from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pathlib import Path


infer_helper = InferenceHelper(dataset='nyu')

#mypath="/hdd2/vol1/deepfakeDatabases/Celeb-DF-v2/cropped_images/Celeb-synthesis/id0_id1_0002/"
#mypath="/hdd2/vol1/deepfakeDatabases/Celeb-DF-v2/cropped_images/Celeb-real/id0_0000"

db_dir="/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/images_test_cropped_1x/"
save_dir="/hdd2/vol2/deepfakeDatabases/Celeb-DF-v2/depth_estimation_AdaBins_test_cropped_1x"

count = 0

for f in listdir(db_dir):
    #Celeb-real, Celeb-Synthesis...
    Path(save_dir).mkdir(parents=True, exist_ok=True)  
    #dejanski file
    if isfile(join(db_dir, f)):
        if count%100==0:
            print("Smo tule: " +  str(count))
        save_path=join(save_dir, f)
        # predict depth of a single pillow image
        img = Image.open(join(db_dir, f))  # any rgb pillow image
        size = img.size
        img = img.resize((640,480))
        bin_centers, predicted_depth = infer_helper.predict_pil(img)

        plt.imsave(save_path, predicted_depth[0][0], cmap='plasma')

        img = Image.open(save_path)
        img = img.resize(size)
        img.save(save_path)
        count = count+1
