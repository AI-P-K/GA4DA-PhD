import sys
from traceback import print_exc

import cv2
from torch.utils.data import Dataset
import os
import re
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import torch


def contains_crop(tuple_of_transforms):
    for index, transform in enumerate(tuple_of_transforms):
        # Convert each element to string
        transform_str = str(transform)
        # Check for 'crop' or 'Crop' exactly
        if "crop" in transform_str or "Crop" in transform_str:
            return index
    return -1

def replace_element_in_tuple(tuple_of_transforms, index, new_element):
    if index == -1:
        return tuple_of_transforms  # No replacement needed if index is -1
    # Create a new tuple with the replacement
    new_tuple = tuple_of_transforms[:index] + (new_element,) + tuple_of_transforms[index + 1:]
    return new_tuple

class DatasetSplitter:
    def __init__(self, dataset_folder):
        """
        A class to split a dataset into training and validation sets.

        :param dataset_folder: The path to the folder containing class subfolders.
        """
        self.dataset_folder = dataset_folder
        self.classes_names = sorted(os.listdir(dataset_folder))


    def get_train_val_filepaths(self, validation_set_size=0.2, random_state=42, n_splits=5):
        """
        Get train and validation image file paths for a 5-fold stratified shuffle split.

        :param validation_set_size: The percentage of images to use for validation (default is 0.2, i.e., 20%).
        :param random_state: Random seed for reproducibility.
        :param n_splits: Number of splits to perform.
        :return: Lists of train and validation image file paths for all splits.
        """
        # Create a list to store train-validation splits
        train_images_filepaths_splits = []
        val_images_filepaths_splits = []

        # Create empty lists to store file paths and class labels
        images_filepaths = []
        class_labels = []

        # Iterate through class subfolders
        for class_idx, class_name in enumerate(self.classes_names):
            class_folder = os.path.join(self.dataset_folder, class_name)

            # Get file paths for images in the current class folder
            class_filepaths = [os.path.join(class_folder, file) for file in sorted(os.listdir(class_folder))]

            # Extend the lists with file paths and assign class labels
            images_filepaths.extend(class_filepaths)
            class_labels.extend([class_idx] * len(class_filepaths))  # Assign labels to samples

        # Initialize StratifiedShuffleSplit with n_splits
        stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits,
                                                     test_size=validation_set_size,
                                                     random_state=random_state)




        # Generate the splits
        for train_indices, val_indices in stratified_splitter.split(images_filepaths, class_labels):
            # Use the indices to create the train and validation sets for this split
            train_images_filepaths = [images_filepaths[i] for i in train_indices]
            val_images_filepaths = [images_filepaths[i] for i in val_indices]
            # Append the splits to the lists
            train_images_filepaths_splits.append(train_images_filepaths)
            val_images_filepaths_splits.append(val_images_filepaths)

        return train_images_filepaths_splits, val_images_filepaths_splits, self.classes_names, len(self.classes_names)

class ClassSpecificTransformationsDataset(Dataset):
    def __init__(self, images_filepaths, classes_names, ga_solution_transforms, input_size, required_transforms=None, save_path=None):
        """
        Custom dataset class to handle images with class-specific transformations.

        :param images_filepaths: List of filepaths to the images.
        :param classes_names: List of class names corresponding to each image.
        :param augmentation_transforms: List of augmentation transformations to apply.
        :param input_size: Size of the input image (assumed to be square).
        :param transforms: Optional, additional transformations to apply to the images.
        """
        self.images_filepaths = images_filepaths
        self.classes_names = classes_names
        self.ga_solution_transforms = ga_solution_transforms
        self.transforms = required_transforms
        self.input_size = input_size
        self.save_path = save_path

    def __len__(self):
        """
        Get the number of images in the dataset.

        :return: The number of images in the dataset.
        """
        return len(self.images_filepaths)

    def random_visualize_data(self, image_orig_vis, image, label, save_path):
        transforms_names = [str(x).split('(')[0] for x in self.ga_solution_transforms]
        image_vis = image.permute(1, 2, 0).numpy()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        unnormalized_image = (image_vis * std) + mean
        unnormalized_image = np.clip(unnormalized_image, 0, 1)
        visulize_transforms = A.Compose([
            A.Resize(self.input_size, self.input_size),
        ])
        image_orig_vis = visulize_transforms(image=image_orig_vis)['image']
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_orig_vis)
        axes[0].axis('off')  # Hide the axis
        axes[0].set_title(f"Original")
        # Display the second image on the right
        axes[1].imshow(unnormalized_image)
        axes[1].axis('off')  # Hide the axis
        axes[1].set_title(f"{transforms_names[label]}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{transforms_names[label]}.png'))
        plt.clf()
        plt.close()

    def __getitem__(self, idx, visualize=None):
        """
        Get the image and its corresponding label at the given index.

        :param idx: Index of the image to retrieve.
        :return: The preprocessed image and its corresponding label.
        """
        image_filepath = self.images_filepaths[idx]
        class_name = os.path.normpath(image_filepath).split(os.sep)[-2]  # Extract class name from the image path
        label = sorted(self.classes_names).index(class_name)  # Get the index of the class name in the list of classes
        original_image = Image.open(image_filepath).convert('RGBA')  # Open and convert the image to RGBA mode
        img_width, img_height = original_image.size

        crop_ratio = 0.9
        image = Image.new('RGB', original_image.size, 'WHITE')  # Create a new RGB image with a white background
        image.paste(original_image, (0, 0), original_image)  # Paste the original image onto the new RGB image
        image.convert('RGB')  # Convert the image to RGB mode
        image = np.array(image)  # Convert the image to a NumPy array
        image_orig_vis = image

        if self.transforms is not None:
            # If additional transforms are provided, apply them to the image
            image = self.transforms(image=image)['image']
        elif self.ga_solution_transforms is not None:
            # If no additional transforms are provided, apply class-specific augmentation transforms
            index_with_crop = contains_crop(self.ga_solution_transforms)
            if index_with_crop != -1:
                if "CenterCrop" in str(self.ga_solution_transforms[index_with_crop]):
                    # print(f"H, W:{img_height, img_width}; CenterCrop H, W:{round(img_height*crop_ratio)}, {round(img_width*crop_ratio)}")
                    new_element = A.CenterCrop(height=round(img_height*crop_ratio), width=round(img_width*crop_ratio), always_apply=True, p=1.0)
                    self.ga_solution_transforms = replace_element_in_tuple(self.ga_solution_transforms, index_with_crop, new_element)
                elif "RandomCrop" in str(self.ga_solution_transforms[index_with_crop]):
                    # print(f"H, W:{img_height, img_width}; RandomCrop H, W:{round(img_height * crop_ratio)}, {round(img_width * crop_ratio)}")
                    new_element = A.RandomCrop(height=round(img_height*crop_ratio), width=round(img_width*crop_ratio), always_apply=True, p=1.0)
                    self.ga_solution_transforms = replace_element_in_tuple(self.ga_solution_transforms, index_with_crop,
                                                                           new_element)
                elif "RandomResizedCrop" in str(self.ga_solution_transforms[index_with_crop]):
                    new_element = A.RandomResizedCrop(height=round(img_height*crop_ratio), width=round(img_width*crop_ratio), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=True, p=1.0)
                    self.ga_solution_transforms = replace_element_in_tuple(self.ga_solution_transforms, index_with_crop,
                                                                           new_element)
                elif "Crop" in str(self.ga_solution_transforms[index_with_crop]):
                    cropped_width = int(img_width * crop_ratio)
                    cropped_height = int(img_height * crop_ratio)
                    x_min = (img_width - cropped_width) // 2
                    y_min = (img_height - cropped_height) // 2
                    x_max = x_min + cropped_width
                    y_max = y_min + cropped_height
                    new_element = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True, p=1.0)
                    self.ga_solution_transforms = replace_element_in_tuple(self.ga_solution_transforms, index_with_crop,
                                                                           new_element)

            cs_transforms = A.Compose([
                self.ga_solution_transforms[label],
                A.Resize(self.input_size, self.input_size),
                A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(always_apply=True, p=1.0),
            ])

            try:
                image = cs_transforms(image=image)['image']
            except:
                print('Reason for fail', cs_transforms)

            sanity_check_transforms = A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(always_apply=True, p=1.0),
            ])
            sanity_check_img = sanity_check_transforms(image=image_orig_vis)['image']

            counter = 0
            max_attempts = 100
            while torch.equal(sanity_check_img, image) and counter < max_attempts:
                cs_transforms = A.Compose([
                    self.ga_solution_transforms[label],
                    A.Resize(self.input_size, self.input_size),
                    A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(always_apply=True, p=1.0),
                ])
                image = cs_transforms(image=image_orig_vis)['image']
                counter += 1
                if max_attempts == 99:
                    print('Transformation does not work: ', self.ga_solution_transforms[label])
            if visualize:
                self.random_visualize_data(image_orig_vis, image, label, self.save_path)
        return image, label


class AlbumentationsImageTransformations:
    def __init__(self, input_size):
        """
        A class to define the required image transformations.

        :param input_size: The size to which the images will be resized (assumed to be square).
        """
        self.input_size = input_size

    def get_baseline_global_augmentation(self, randomGlobalAugmentation):
        baseline_global_transformations = A.Compose([
            randomGlobalAugmentation,
            A.Resize(height=self.input_size, width=self.input_size, always_apply=True),
            A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return baseline_global_transformations

    def get_transforms(self):
        """
        Get the required image transformations.

        :return: An instance of the required image transformations.
        """
        required_image_transformations = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(always_apply=True, p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        return required_image_transformations



