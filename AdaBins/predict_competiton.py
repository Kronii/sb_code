from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pathlib import Path


infer_helper = InferenceHelper(dataset='nyu')

#mypath="/hdd2/vol1/deepfakeDatabases/Celeb-DF-v2/cropped_images/Celeb-synthesis/id0_id1_0002/"
#mypath="/hdd2/vol1/deepfakeDatabases/Celeb-DF-v2/cropped_images/Celeb-real/id0_0000"

db_dir="/hdd2/vol1/deepfakeDatabases/cropped_videos/Celeb-DF-v2/train/faces"
save_dir="/hdd2/vol1/deepfakeDatabases/dfgc/depth_estimation_AdaBins"

count = 0

for dir2 in listdir(db_dir):
    #id_0001...
    mypath = join(db_dir, dir2)
    for f in listdir(mypath):
        Path(join(save_dir, dir2)).mkdir(parents=True, exist_ok=True)  
        #dejanski file
        if isfile(join(mypath, f)):
            if count%100==0:
                print("Smo tule: " +  str(count))
            save_path=join(join(save_dir, dir2), f)
            # predict depth of a single pillow image
            img = Image.open(join(mypath, f))  # any rgb pillow image
            size = img.size
            img = img.resize((640,480))
            bin_centers, predicted_depth = infer_helper.predict_pil(img)

            plt.imsave(save_path, predicted_depth[0][0], cmap='plasma')

            img = Image.open(save_path)
            img = img.resize(size)
            img.save(save_path)
            count = count+1
