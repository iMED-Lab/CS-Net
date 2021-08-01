<center>
  <img src="http://ww1.sinaimg.cn/large/005CmS3Mgy1gt1du2q621j609a02wq3d02.jpg" 
</center>

# CS2-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging

Implementation of [CS2-Net MedIA 2020](https://www.sciencedirect.com/science/article/pii/S1361841520302383)

---

## Overview

<center>
  <img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gk9zy0quvwj319g0ridpg.jpg" height="60%" width="60%"
</center>

## Requirements

![](https://img.shields.io/badge/PyTorch-0.4.1-orange)  ![](https://img.shields.io/badge/visdom-0.2.0-orange)  ![](https://img.shields.io/badge/SimpleITK-latest-orange)    

## Get Started

- ```train3d.py``` is used to train the 3D segmentation network.

- ```predict3d.py``` is used to test the trained model.

- Please note that you should change the dataloader definition in ```train3d.py```.

## Examples

- MRA brain vessel segmentation

<center>
  <img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gka00rmmqgj31i60mu1kx.jpg" height="60%" width="60%"/>
</center>

- Synthetic & VascuSynth

<center>
  <img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gka01tpjvzj30xu0r2b1l.jpg" height="60%" width="60%"/>
</center>

## Citation

> @article{mou2020cs2,
> title={CS2-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging},
> author={Mou, Lei and Zhao, Yitian and Fu, Huazhu and Liux, Yonghuai and Cheng, Jun and Zheng, Yalin and Su, Pan and Yang, Jianlong and Chen, Li and Frangi, Alejandro F and others},
> journal={Medical Image Analysis},
> pages={101874},
> year={2020},
> publisher={Elsevier}
> }

#### Corrections to: CS2-Net- Deep learning segmentation of curvilinear structures in medical imaging

The original comparison results in Table 8 on page 14 are:

<img src="http://ww1.sinaimg.cn/mw690/005CmS3Mly1gki44l3iojj30n205s759.jpg" width =350 align=center/>

The corrected comparison results are:

<img src="http://ww1.sinaimg.cn/mw690/005CmS3Mly1gki47buiedj30os05u75k.jpg" width =350 align=center/>

## Useful Links

| DRIVE          | http://www.isi.uu.nl/Research/Databases/DRIVE/              |
| :------------- | :---------------------------------------------------------- |
| **STARE**      | **http://www.ces.clemson.edu/ahoover/stare/**               |
| **IOSTAR**     | **http://www.retinacheck.org/**                             |
| **ToF MIDAS**  | **http://insight-journal.org/midas/community/view/21**      |
| **Synthetic**  | **https://github.com/giesekow/deepvesselnet/wiki/Datasets** |
| **VascuSynth** | **http://vascusynth.cs.sfu.ca/Data.html**                   |

