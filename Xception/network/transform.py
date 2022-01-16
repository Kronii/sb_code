from torchvision import transforms
import random

import cv2
import numpy as np

xception_transforms_train = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
    #transforms.Resize((299, 299)),
    #transforms.RandomRotation((-30, 30)),
    #transforms.RandomHorizontalFlip(0.3),
    #transforms.ColorJitter(brightness=(0.1, 0.4), contrast=(0.1, 0.4), saturation=(0.1, 0.4), hue=(0.1, 0.4)),
    #transforms.RandomPerspective(0.5, 0.2),
    #transforms.RandomGrayscale(p=0.3),
    #transforms.GaussianBlur(kernel_size=3),
    #transforms.ToTensor(),
    #transforms.RandomErasing(p=0.2),
    #transforms.Normalize([0.5] * 3, [0.5] * 3)
])

xception_transforms_validation = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])