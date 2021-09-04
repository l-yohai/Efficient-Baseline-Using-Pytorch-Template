# Efficient baseline using pytorch template

This pytorch template is for Image Classification of AI Competition.
Diffrence between original pytorch-template and efficient baseline is as follows.

- Can use hard splitted valid_data_loader
- Can add logs in wandb
- Add custom_dataset & custom_valid_dataset
- Can use albumentations not transforms in torchvision
- Can see Confusion Matrix figure in tensorboard
- Can change pretrained model (in timm) simply change to argument of config.

## Requirements
```
Python >= 3.5 (3.6 recommended)
PyTorch >= 0.4 (1.2 recommended)
tqdm (Optional for test.py)
tensorboard >= 1.14 (see Tensorboard Visualization)
wandb >= 0.12.1 (see Wandb Visualization)
sklearn >= 0.24.2
matplotlib >= 3.2.1
seaborn >= 0.11.2
timm >= 0.4.12
```

## Causions

1. ***You must change TODO***
    ```shell
    custom_dataset.py # dataset path
    data_loader.py # augmentations
    model.py # if you use other model that is not in timm
    train.py # init your wandb account
    ```

2. ***And this code uses '.csv' file. Csv file should have image's path and label.***

## Appendix

![wandb](images/wandb.png)

![tensorboard](images/tensorboard.png)

![confusion_matrix](images/confusion_matrix.png)

## Reference

[Pytorch Custom Dataset Examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas)