# I	CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

This repo is the official implementation of [CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_80).

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82eztkb0wj31e40u0axe.jpg" height="60%" width="60%"
</center>



The main contribution of this work is the publication of two scarce datasets in the medical image field.  Plesae click the link below to access the details and source data.

[![](https://img.shields.io/badge/Download-CORN--1-green)](http://www.imed-lab.com/?p=16073) 

## I-1	Experiment Results

#### I-1.1	Vessel Segmentation on Fundus

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f22kgj6j315t0lv1f7.jpg" height="65%" width="65%"/>
</center>



#### I-1.2	Vessel Segmentation on OCT-A images

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f2qvdw5j31ew0brke8.jpg" height="65%" width="65%" />
</center>



#### I-1.3	Nerve fibre tracing on CCM

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f39bqm7j31g70lxqnb.jpg" height="65%" width="65%"/>
</center>


## I-2	Usage:

Use the ```train.py``` and ```predict.py``` to train and test the model on your own dataset, respectively.

## I-3	Requirements:

- PyTorch >= 0.4.1
- tqdm
- cv2
- visdom
- sklearn

The attention module was implemented based on [DANet](https://github.com/junfu1115/DANet). The difference between the proposed module and the original block is that  we added a new 1x3 and 3x1 kernel convolution layer into spatial attention module. Plese refer to the paper for details.

#### I-4	Citation

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



# II	CS$^2$-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging

The extension of the 2D CS-Net

<img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gjpsnhr4gyj31900to7el.jpg" alt="iShot2020-10-15 10.07.46.png" style="zoom:40%;" />

## II-1	3D Volume Segmentation Results

#### II-1.1	MRA Brain Vessel

<img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gjpsreqltmj30ni0a8gs6.jpg" alt="iShot2020-10-15 10.13.00.png" style="zoom:80%;" />

#### II-1.2	Synthetic & VascuSynth

<img src="http://ww1.sinaimg.cn/mw690/005CmS3Mly1gjpst76lklj30ne0iiwsm.jpg" alt="iShot2020-10-15 10.14.38.png" style="zoom:80%;" />

## II-2	Usage:

```train3d.py``` to train the 3D segmentation network.

```predict3d.py``` is used to test the trained model.

Please note that you should change the dataloader definition in ```train3d.py```.

### II-2.1

Requirements:

- PyTorch = 0.4.1
- visdom
- SimpleITK: 
  - ```pip install SimpleITK```



### II-3	Citation

...

## III	Dataset Links:

1. DRIVE: http://www.isi.uu.nl/Research/Databases/DRIVE/
2. STARE: http://www.ces.clemson.edu/ahoover/stare/
3. IOSTAR: http://www.retinacheck.org/
4. ToF MIDAS: http://hdl.handle.net/1926/594
5. Synthetic: https://github.com/giesekow/deepvesselnet/wiki/Datasets
6. VascuSyynth: http://vascusynth.cs.sfu.ca/Data.html


