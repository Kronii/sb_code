
from facenet_pytorch import MTCNN

mtcnn = MTCNN(device='cuda:0').eval()

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


def read_crop_face(image_path):
    image = cv2.imread(image_path)[:, :, ::-1]
    height, width = image.shape[:2]
    imageArray = Image.fromarray(image)

    boxes, _ = mtcnn.detect(imageArray)
    x, y, size = get_boundingbox(boxes.flatten(), width, height)
    cropped_face = image[y:y + size, x:x + size]

    # except:
    #     print(image_path)

    return cropped_face


if __name__ == '__main__':
    # Modify the following directories to yourselves
    #VIDEO_ROOT = '../../DeepTomCruise/'          # The base dir of CelebDF-v2 dataset
    #OUTPUT_PATH = '../../DeepTomCruiseBoth_patched/'    # Where to save cropped training faces
    #XT_PATH = "../tom-list.txt"    # the given train-list.txt file

    PICTURES_ROOT = '../../Celeb-DF-v2/'          # The base dir of CelebDF-v2 dataset
    OUTPUT_PATH = '../../Celeb-DF-v2-test/'    # Where to save cropped training faces

    data=os.listdir(PICTURES_ROOT)
    print(data)
    # Face detector
    mtcnn = MTCNN(device='cuda:0').eval()
    for line in data:
        video_name = line[2:-1]
        video_path = os.path.join(VIDEO_ROOT, video_name)
        save_dir = OUTPUT_PATH + video_name.split('.')[0]
        get_face_patch(video_path, save_dir)