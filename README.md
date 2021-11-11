# CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

Implementation of [CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_80)

For the details of 3D extended version of CS-Net, please refer to [CS2-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging](cs2net.md)

---

## Overview

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82eztkb0wj31e40u0axe.jpg" height="60%" width="60%"
</center>

The main contribution of this work is the publication of two scarce datasets in the medical image field.  Plesae click the link below to access the details and source data. [![](https://img.shields.io/badge/Download-CORN--1-green)](http://www.imed-lab.com/?p=16073)

## Requirements

![](https://img.shields.io/badge/PyTorch-%3E%3D0.4.1-orange)  ![](https://img.shields.io/badge/tqdm-latest-orange)  ![](https://img.shields.io/badge/cv2-latest-orange)  ![](https://img.shields.io/badge/visdom-%3E%3D0.2.0-orange)  ![](https://img.shields.io/badge/sklearn-latest-orange)  

The attention module was implemented based on [DANet](https://github.com/junfu1115/DANet). The difference between the proposed module and the original block is that  we added a new 1x3 and 3x1 kernel convolution layer into spatial attention module. Plese refer to the paper for details.

## Get Started

Using the ```train.py``` and ```predict.py``` to train and test the model on your own dataset, respectively.

## Examples

- Vessel segmentation on Fundus

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f22kgj6j315t0lv1f7.jpg" height="60%" width="60%"/>
</center>

- Vessel segmentation on OCT-A images

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f2qvdw5j31ew0brke8.jpg" height="60%" width="60%" />
</center>

- Nerve fiber tracing on CCM

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f39bqm7j31g70lxqnb.jpg" height="60%" width="60%"/>
</center>

## Citation

```
@inproceedings{mou2019cs,
title={CS-Net: channel and spatial attention network for curvilinear structure segmentation},
author={Mou, Lei and Zhao, Yitian and Chen, Li and Cheng, Jun and Gu, Zaiwang and Hao, Huaying and Qi, Hong and Zheng, Yalin and Frangi, Alejandro and Liu, Jiang},
booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
pages={721--730},
year={2019},
organization={Springer}
}
```



## Useful Links

| DRIVE          | http://www.isi.uu.nl/Research/Databases/DRIVE/              |
| :------------- | :---------------------------------------------------------- |
| **STARE**      | **http://www.ces.clemson.edu/ahoover/stare/**               |
| **IOSTAR**     | **http://www.retinacheck.org/**                             |
| **ToF MIDAS**  | **http://insight-journal.org/midas/community/view/21**      |
| **Synthetic**  | **https://github.com/giesekow/deepvesselnet/wiki/Datasets** |
| **VascuSynth** | **http://vascusynth.cs.sfu.ca/Data.html**                   |
