import os
import cv2
from facenet_pytorch import MTCNN, extract_face
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import subprocess
import json
import dlib
from imutils import face_utils
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)


def select_biggest(face_list):
    max_face = 0
    ix = 0
    for i, face in enumerate(face_list):
        if face.area() > max_face:
            ix = i
            max_face = face.area()
    return ix


def get_i_face(videoPath, save_root, video_type):
    frame_types = get_frame_types(videoPath)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor("../../shape_predictor_68_face_landmarks.dat")
    if i_frames:
        basename = os.path.splitext(os.path.basename(videoPath))[0]
        v_cap = cv2.VideoCapture(videoPath)
        for frame_no in i_frames:
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = v_cap.read()

            height, width = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                #faces = face_detector(image, 1)
                print("used mtcnn" + str(frame_no))
                faces, _ = mtcnn.detect(image)
                faces = faces.flatten()
                faces = [dlib.rectangle(int(faces[0]), int(faces[1]), int(faces[2]), int(faces[3]))]

                biggest = faces[0]
                if np.size(faces) > 1:
                    biggest = faces[select_biggest(faces)]
                box = [int(biggest.left()), int(biggest.top()), int(biggest.right()), int(biggest.bottom())]
                
                #Draw boxes and save faces
                #image_draw = Image.fromarray(image)
                face = biggest
                # Make the prediction and transfom it to numpy array
                shape = landmark_detector(image, face)
                shape = face_utils.shape_to_np(shape)
                #print(biggest.tolist())
                landms=[]
                x1 = face.left()
                y1 = face.top()
                w = face.right() - x1
                h = face.bottom() - y1
                # draw box over face
                #cv2.rectangle(image, (x1,y1), (x1+w,y1+h), (0,255,0), 2)

                # Draw on our image, all the finded cordinate points (x,y) 
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0))
                    #cv2.rectangle(image, (x-24,y-24), (x+24,y+24), (0,255,0), 1) #   y - 24:y + 24, x - 24:x + 24]
                    landms.append(int(x))
                    landms.append(int(y))

                #json_object = {'box':box,'landms':landms}

                #d[video_type+'_'+basename+'_'+str(frame_no)]=json_object

                cv2.imwrite('faces/new_annotated_face_{}.png'.format(frame_no), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))                
                
                outname = save_root+video_type+'_'+basename+'_'+str(frame_no)+'.png'
                #cv2.imwrite(outname, frame)

            except:
                print('no face: ' + videoPath)
                outname = save_root+'no_face_'+video_type+'_'+basename+'_'+str(frame_no)+'.png'
                #cv2.imwrite(outname, frame)


        v_cap.release()
    else:
        print ('No I-frames in '+videoPath)   


def get_face_test(videoPath, save_root, video_type):
    select_nums = 30
    frame_types = get_frame_types(videoPath)
    #i_frames = [x[0] for x in frame_types if x[1]=='I']
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len > select_nums:
        samples = np.linspace(0, v_len - 1, select_nums).round().astype(int)
    else:
        samples = np.linspace(0, v_len - 1, v_len).round().astype(int)
    #basename = os.path.splitext(os.path.basename(videoPath))[0]
    print(samples)
    for frame_no in range(v_len):
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = v_cap.read()
        if frame_no in samples:
            path_split = videoPath.split('/')
            person_number = path_split[len(path_split)-2]
            video_number = path_split[len(path_split)-1][:-4]
            
            height, width = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                faces, _ = mtcnn.detect(image)
                faces = faces.flatten()
                faces = [dlib.rectangle(int(faces[0]), int(faces[1]), int(faces[2]), int(faces[3]))]

                biggest = faces[0]
                if np.size(faces) > 1:
                    biggest = faces[select_biggest(faces)]
                box = [int(biggest.left()), int(biggest.top()), int(biggest.right()), int(biggest.bottom())]

                json_object = {'box':box}

                d[video_type+'_'+person_number+'_'+video_number+'_'+str(frame_no)]=json_object

                #cv2.imwrite('faces/annotated_face_{}.png'.format(frame_no), image)   
                            
                
                #outname = save_root+video_type+'_'+person_number+'_'+video_number+'_'+str(frame_no)+'.png'
                #cv2.imwrite(outname, frame)

            except:
                print('no face: ' + videoPath)
                os.path.join(save_root, "no_face")
                outname = save_root+'no_face_'+video_type+'_'+person_number+'_'+video_number+'_'+str(frame_no)+'.png'
                cv2.imwrite(outname, frame)
    
    v_cap.release() 


def get_face(videoPath, save_root, select_nums=10):
    numFrame = 0
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len > select_nums:
        samples = np.linspace(0, v_len - 1, 10).round().astype(int)
    else:
        samples = np.linspace(0, v_len - 1, v_len).round().astype(int)
    samples = np.linspace(0, v_len - 1, v_len).round().astype(int)
    for j in range(v_len):
        success, vframe = v_cap.read()
        if j in samples:
            height, width = vframe.shape[:2]
            image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            try:
                boxes, _ = mtcnn.detect(image)
                x, y, size = get_boundingbox(boxes.flatten(), width, height)
                cropped_face = vframe[y:y + size, x:x + size]

                s = str(numFrame)
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                #cv2.imwrite(os.path.join(save_root, "%s.png") % s, cropped_face)
                cv2.imwrite(os.path.join(save_root, "%s.png") % s, vframe)
                numFrame += 1

            except:
                print(videoPath)
    v_cap.release()


if __name__ == '__main__':
    # Modify the following directories to yourselves
    VIDEO_ROOT = '/hdd/deepfakeDatabases/Celeb-DF-v2/original_videos/'          # The base dir of CelebDF-v2 dataset
    OUTPUT_PATH = '/hdd/deepfakeDatabases/Celeb-DF-v2/dlib_test/'    # Where to save cropped training faces
    TXT_PATH = "/hdd/deepfakeDatabases/Celeb-DF-v2/original_videos/List_of_testing_videos.txt"   # the given train-list.txt file

    with open(TXT_PATH, "r") as f:
        data = f.readlines()
    d={}

    # Face detector
    mtcnn = MTCNN(device='cuda:0').eval() #mtcnn = MTCNN(keep_all=True)
    for line in data:
        video_name = line[2:-1]
        video_type = line[0:1]
        video_path = os.path.join(VIDEO_ROOT, video_name)
        #save_dir = OUTPUT_PATH + video_name.split('.')[0]
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        save = OUTPUT_PATH
        get_i_face(video_path, save, video_type)
    with open('Celeb-DF-v2-test.json', 'w') as f:
        json.dump(d, f, indent=4)
