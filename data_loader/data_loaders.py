from torchvision import datasets, transforms
from base import BaseDataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from custom_dataset import *


# TODO: Edit augmentations you want
transform_list = [
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=224, width=224, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ])
]


class CustomDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        # TODO: Edit augmentations you want
        self.transform = A.Compose([
            A.OneOf(transform_list, p=0.5),
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        self.dataset = CustomDatasetFromImages(
            self.data_dir, self.csv_path, self.transform, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CustomValidDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=False):
        # TODO: Edit augmentations you want
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.dataset = CustomValidDatasetFromImages(
            self.data_dir, self.csv_path, self.transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
