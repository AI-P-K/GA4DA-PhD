# RandomCrop
# MultiplicativeNoise
# RandomScale
# CenterCrop
# Crop
# RandomResizedCrop
import sys

import albumentations as A
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import cv2
from albumentations.pytorch import ToTensorV2


crop_ratio = 0.9

image_filepath = 'Datasets/BrainTumorMRI/notumor/Tr-no_1467.jpg'
class_name = os.path.normpath(image_filepath).split(os.sep)[-2]  # Extract class name from the image path
original_image = Image.open(image_filepath).convert('RGBA')  # Open and convert the image to RGBA mode
img_width, img_height = original_image.size
print(img_height, img_width)
cropped_width = int(img_width * crop_ratio)
cropped_height = int(img_height * crop_ratio)
xmin = (img_width - cropped_width) // 2
ymin = (img_height - cropped_height) // 2
xmax = xmin + cropped_width
ymax = ymin + cropped_height
transforms = A.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, always_apply=True, p=1.0)
input_size = 150

cs_transforms = A.Compose([
                A.RandomCrop(height=round(img_height*crop_ratio), width=round(img_width*crop_ratio), always_apply=True, p=1.0),
                A.Resize(input_size, input_size),
                A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(always_apply=True, p=1.0),
            ])




print(original_image)
# sys.exit()
image = Image.new('RGB', original_image.size, 'WHITE')  # Create a new RGB image with a white background
image.paste(original_image, (0, 0), original_image)  # Paste the original image onto the new RGB image
image.convert('RGB')  # Convert the image to RGB mode
image = np.array(image)
image = cs_transforms(image=image)['image']
sys.exit()
print(image.shape)
plt.imshow(image)
plt.show()


