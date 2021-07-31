# CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

This repo is the official implementation of [CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_80).

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82eztkb0wj31e40u0axe.jpg" height="60%" width="60%"
</center>



The main contribution of this work is the publication of two scarce datasets in the medical image field.  Plesae click the link below to access the details and source data.

[![](https://img.shields.io/badge/Download-CORN--1-green)](http://www.imed-lab.com/?p=16073) 

## Experiment Results

#### 1. Vessel Segmentation on Fundus

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f22kgj6j315t0lv1f7.jpg" height="65%" width="65%"/>
</center>



#### 2. Vessel Segmentation on OCT-A images

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f2qvdw5j31ew0brke8.jpg" height="65%" width="65%" />
</center>



#### 3. Nerve fibre tracing on CCM

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f39bqm7j31g70lxqnb.jpg" height="65%" width="65%"/>
</center>


## Usage:

Using the ```train.py``` and ```predict.py``` to train and test the model on your own dataset, respectively.

## Requirements:

- PyTorch >= 0.4.1
- tqdm
- cv2
- visdom
- sklearn

The attention module was implemented based on [DANet](https://github.com/junfu1115/DANet). The difference between the proposed module and the original block is that  we added a new ![](http://latex.codecogs.com/svg.latex?1\times 3) and ![](http://latex.codecogs.com/svg.latex?3\times 1) kernel convolution layer into spatial attention module. Plese refer to the paper for details.

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



---

# CS2​-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging

The extension of the 2D CS-Net: [CS2​-Net](https://www.sciencedirect.com/science/article/pii/S1361841520302383)

<img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gk9zy0quvwj319g0ridpg.jpg" width =300 align=center/>


## 3D Volume Segmentation Results

#### 1 MRA Brain Vessel 

<img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gka00rmmqgj31i60mu1kx.jpg" width =300 align=center/>


#### 2 Synthetic & VascuSynth

<img src="http://ww1.sinaimg.cn/large/005CmS3Mly1gka01tpjvzj30xu0r2b1l.jpg" width =300 align=center/>


## Usage:

```train3d.py``` is used to train the 3D segmentation network.

```predict3d.py``` is used to test the trained model.

Please note that you should change the dataloader definition in ```train3d.py```.

## Requirements:

- PyTorch = 0.4.1
- visdom
- SimpleITK: 

  > ```pip install SimpleITK```



### Citation

> ```
> @article{mou2020cs2,
>   title={CS2-Net: Deep Learning Segmentation of Curvilinear Structures in Medical Imaging},
>   author={Mou, Lei and Zhao, Yitian and Fu, Huazhu and Liux, Yonghuai and Cheng, Jun and Zheng, Yalin and Su, Pan and Yang, Jianlong and Chen, Li and Frangi, Alejandro F and others},
>   journal={Medical Image Analysis},
>   pages={101874},
>   year={2020},
>   publisher={Elsevier}
> }
> ```

#### Correction to: CS2-Net- Deep learning segmentation of curvilinear structures in medical imaging

The original comparison results in Table 8 on page 14 are:

<img src="http://ww1.sinaimg.cn/mw690/005CmS3Mly1gki44l3iojj30n205s759.jpg" width =350 align=center/>

The corrected comparison results are:

<img src="http://ww1.sinaimg.cn/mw690/005CmS3Mly1gki47buiedj30os05u75k.jpg" width =350 align=center/>



## Dataset Links:

| DRIVE          | http://www.isi.uu.nl/Research/Databases/DRIVE/              |
| :------------- | :---------------------------------------------------------- |
| **STARE**      | **http://www.ces.clemson.edu/ahoover/stare/**               |
| **IOSTAR**     | **http://www.retinacheck.org/**                             |
| **ToF MIDAS**  | **http://insight-journal.org/midas/community/view/21**      |
| **Synthetic**  | **https://github.com/giesekow/deepvesselnet/wiki/Datasets** |
| **VascuSynth** | **http://vascusynth.cs.sfu.ca/Data.html**                   |

