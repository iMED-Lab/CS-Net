from __future__ import print_function, division
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
import SimpleITK as sitk
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
    groundtruth_path = os.path.join(root_dir, sub_dir, 'mesh_label')

    for file in glob.glob(os.path.join(images_path, '*.mha')):
        image_name = os.path.basename(file)[:-8]
        groundtruth_name = image_name + '.mha'

        images.append(file)
        groundtruth.append(os.path.join(groundtruth_path, groundtruth_name))

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

        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image).astype(np.float32)  # [x,y,z] -> [z,y,x]

        label = sitk.ReadImage(gt_path)
        # if use CE loss, type: astype(np.int64), or use MSE type: astype(np.float32)
        label = sitk.GetArrayFromImage(label).astype(np.int64)  # [x,y,z] -> [z,y,x]

        image, label = self.RandomCrop(image, label, crop_factor=(64, 104, 112))  # [z,y,x]

        if self.train:
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        else:
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        image = image / 255
        label = label // 255

        return image, label
