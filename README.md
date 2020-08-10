# CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation

This repo is the official implementation of [CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_80).

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82eztkb0wj31e40u0axe.jpg" height="60%" width="60%"
</center>



The main contribution of this work is the publication of two scarce datasets in the medical image field.  Plesae click the link below to access the details and source data.

[DOWNLOAD](http://www.imed-lab.com/?p=16073) 

## Experiment Results

#### Vessel Segmentation on Fundus

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f22kgj6j315t0lv1f7.jpg" height="65%" width="65%"/>
</center>



#### Vessel Segmentation on OCT-A images

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f2qvdw5j31ew0brke8.jpg" height="65%" width="65%" />
</center>



#### Nerve fibre tracing on CCM

<center>
  <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g82f39bqm7j31g70lxqnb.jpg" height="65%" width="65%"/>
</center>



## Note:

**Requirements**:

- PyTorch >= 0.4.1

- tqdm
- cv2
- visdom
- sklearn

The attention module was implemented based on [DANet](https://github.com/junfu1115/DANet). The difference between the proposed module and the original block is that  we added a new 1x3 and 3x1 kernel convolution layer into spatial attention module. Plese refer to the paper for details.

**Welcome Any Problems of This Project**

