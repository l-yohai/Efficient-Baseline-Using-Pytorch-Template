from albumentations.augmentations.transforms import HorizontalFlip
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform, train=True):
        self.train = train

        # TODO: Edit your Test dataset path
        self.data_dir = data_dir if self.train else "YOUR_TEST_DATA_DIR"
        # TODO: Edit your Test csv file path
        self.csv_path = csv_path if self.train else "YOUR_TEST_CSV_PATH"

        # Transforms with albumentations
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path)

        # TODO: Edit your image path
        # If you store image path as absolute path in csv, erase self.data_dir
        self.image_arr = np.asarray(self.data_dir + self.data_info['path'])
        # Train Dataset
        if self.train:
            # TODO: Edit your label column
            self.label_arr = np.asarray(self.data_info['label'])

        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        if self.train:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            transform_image = self.transform(image=img_as_img)
            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.label_arr[index]
            return (transform_image['image'], single_image_label)
        else:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            if self.transform is not None:
                transform_image = self.transform(image=img_as_img)
            return transform_image['image']

    def __len__(self):
        return self.data_len


class CustomValidDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform):
        self.data_dir = data_dir
        self.csv_path = csv_path

        self.transform = transform
        self.data_info = pd.read_csv(self.csv_path)

        # TODO: Edit your image path
        # If you store image path as absolute path in csv, erase self.data_dir
        self.image_arr = np.asarray(self.data_dir + self.data_info['path'])
        # TODO: Edit your label column
        self.label_arr = np.asarray(self.data_info['label'])
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = np.array(Image.open(single_image_name))

        transform_image = self.transform(image=img_as_img)
        single_image_label = self.label_arr[index]

        return (transform_image['image'], single_image_label)

    def __len__(self):
        return self.data_len
