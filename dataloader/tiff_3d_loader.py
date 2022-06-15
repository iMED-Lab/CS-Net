from __future__ import print_function, division
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
from skimage import io
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter

warnings.filterwarnings('ignore')


def load_dataset(root_dir, train=True):
    images = []
    groundtruth = []
    if train:
        sub_dir = 'training'
    else:
        sub_dir = 'test'
    images_path = os.path.join(root_dir, sub_dir, 'images')
    groundtruth_path = os.path.join(root_dir, sub_dir, 'ground_truth')

    for file in glob.glob(os.path.join(images_path, '*.tif')):
        images.append(file)
        groundtruth.append(file.replace(images_path, groundtruth_path))

    return images, groundtruth


class Data(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=512):

        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train)

    def __len__(self):
        return len(self.images)

    def RandomCrop(self, image, label, crop_factor=(0, 0, 0)):
        """
        Make a random crop of the whole volume
        :param image:
        :param label:
        :param crop_factor: The crop size that you want to crop
        :return:
        """
        w, h, d = image.shape
        z = random.randint(0, w - crop_factor[0])
        y = random.randint(0, h - crop_factor[1])
        x = random.randint(0, d - crop_factor[2])

        image = image[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
        label = label[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
        return image, label

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]
        image = io.imread(img_path)
        label = io.imread(gt_path)

        image, label = self.RandomCrop(image, label, crop_factor=(64, 104, 112))  # [z,y,x]

        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        image = image / image.max()
        label = label / image.max()

        return image, label
