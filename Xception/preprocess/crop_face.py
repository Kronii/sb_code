import os
import cv2
import subprocess
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import numpy as np
import dlib
from imutils import face_utils

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


def get_face(videoPath, save_root, select_nums=10):
    frame_types = get_frame_types(videoPath)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    numFrame = 0
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(v_len):
        success, vframe = v_cap.read()
        if j in i_frames:
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
                cv2.imwrite(os.path.join(save_root, "%s.png") % s, cropped_face)
                numFrame += 1

            except:
                print(videoPath)
    v_cap.release()


def get_face_patch(videoPath, save_root, select_nums=10):
    frame_types = get_frame_types(videoPath)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    numFrame = 0
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmark_detector = dlib.shape_predictor("../../shape_predictor_68_face_landmarks.dat")

    for j in range(v_len):
        success, vframe = v_cap.read()
        if j in i_frames:
            height, width = vframe.shape[:2]
            image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(image)

            try:
                faces, _ = mtcnn.detect(image)
                faces = faces.flatten()
                faces = [dlib.rectangle(int(faces[0]), int(faces[1]), int(faces[2]), int(faces[3]))]
                
                face = biggest
                shape = landmark_detector(image, face)
                shape = face_utils.shape_to_np(shape)

                for ix, (x, y) in enumerate(shape):
                    cropped_face = vframe[y - 24:y + 24, x - 24:x + 24]

                    s = str(numFrame) + '_' + str(ix)
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    cv2.imwrite(os.path.join(save_root, "%s.png") % s, cropped_face)
                    numFrame += 1

            except:
                print(videoPath)
    v_cap.release()


if __name__ == '__main__':
    # Modify the following directories to yourselves
    #VIDEO_ROOT = '../../DeepTomCruise/'          # The base dir of CelebDF-v2 dataset
    #OUTPUT_PATH = '../../DeepTomCruiseBoth_patched/'    # Where to save cropped training faces
    #XT_PATH = "../tom-list.txt"    # the given train-list.txt file

    VIDEO_ROOT = '../../Celeb-DF-v2/'          # The base dir of CelebDF-v2 dataset
    OUTPUT_PATH = '../../Celeb-DF-v2-test/'    # Where to save cropped training faces
    TXT_PATH = "../test-list.txt"    # the given train-list.txt file

    with open(TXT_PATH, "r") as f:
        data = f.readlines()

    # Face detector
    mtcnn = MTCNN(device='cuda:0').eval()
    for line in data:
        video_name = line[2:-1]
        video_path = os.path.join(VIDEO_ROOT, video_name)
        save_dir = OUTPUT_PATH + video_name.split('.')[0]
        get_face_patch(video_path, save_dir)
