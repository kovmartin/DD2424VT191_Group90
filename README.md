# Our main project solution 

<p align="left">
    <a href="#Technical_information">Technical information</a> &bull;
    <a href="#Abstract">Abstract</a> &bull;
    <a href="#Solution_details">Solution details</a> &bull;
    <a href="#Setup">Setup</a> &bull;
    <a href="#Dataset_instructions">Dataset instructions</a>
</p>

<a name="Technical_information"></a>
## Technical information
* Group number: 90
* Members: 

|          |       Dávid Egri       |       Martin Kovács       |
| :------: | :--------------------: | :-----------------------: |
|  email   | egridavid.bp@gmail.com | kovacsmartin095@gmail.com |
| T-number |      950909-T592       |        951211-T395        |

<a name="Abstract"></a>
## Abstract

<p float="center">
  <img src="https://challenge2018.isic-archive.com/wp-content/uploads/2018/03/ISIC_2018.svg" width="100" />

  <img src="https://challenge2018.isic-archive.com/wp-content/uploads/2018/03/banner.jpg" width="750" /> 
</p>

In this repository we summarize our results for the main project work of the DD2424 Deep Learning in Data Science course. We solved the first task of the ISIC 2018 challenge which was to make automated predictions of lesion segmentation boundaries within dermoscopic images. Thus, it was a medical computer vision application where we had to solve a binary semantic segmentation problem. Our goal was to build variants of the famous U-net architecture and we tested them against each other. In all of the models we used the transfer learning approach and as well, tested the effect of the transfer learning too. We tried out different loss functions and eventually came up with a costume one. We did experiments with data augmentation using simple geometric transformations. With these models we could achieve comparable results with the contestants in the competition.

<a name="Solution_details"></a>
## Solution details 

Our work is a Python3 / [Keras](https://github.com/fchollet/keras) implementation based on the paper, based on the idea of the following paper:

[**U-Net: Convolutional Networks for Biomedical Image Segmentation**](https://arxiv.org/abs/1505.04597), by Olaf Ronneberger and Philipp Fischers. (2015) 

We worked in Google Colaboratory with free GPU runtime.


<a name="Setup"></a>



## Setup

To run one of the notebooks you need the following:

- a machine with GPU
- Python3
- Pathlib, Numpy, Keras, Matplotlib, Pandas, skimage and OpenCV2 packages

These conditions are fulfilled in Google Colaboratory. Also, we worked in Google Colaboratory with mounted Google Drive with the following folder structure.
Please note that the main root folder was *My Drive*.

```
./ISIC2018/ISIC2018_Task1-2_Training_Input
./ISIC2018/ISIC2018_Task1_Training_GroundTruth
./Final_solution/models/Unet_VGG16_classic_data_augmentation.ipynb
./Final_solution/models/Unet_VGG16_classic_no_transfer_learning.ipynb
./Final_solution/models/Unet_VGG16_classic.ipynb
./Final_solution/models/Unet_VGG16_dilated_convolutions.ipynb
./Final_solution/models/Unet_VGG16_simplified.ipynb
./Final_solution/__init__.py
./Final_solution/callbacks.py
./Final_solution/data_utils.py
./Final_solution/eval_utils.py
./Final_solution/losses.py
./Final_solution/metrics.py
./Final_solution/model_utils.py
```

<a name="Dataset_instructions"></a>
## Dataset instructions

Please see `DOWNLOAD_DATASET.md`.
