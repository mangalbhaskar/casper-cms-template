/*
Title: Datasets and Data Creation for Training Machines
Decription: Datasets and Data Creation for Training Machines
Author: Bhaskar Mangal
Date: 24th-Jul-2018
Last updated: 24th-Jul-2018
Tags: Datasets and Data Creation for Training Machines
*/

**Table of Contents**

[TOC]

# Datasets and Data Creation for Training Machines


## Creation for Training Machines

### Image Labeling Tools
* https://en.wikipedia.org/wiki/List_of_manual_image_annotation_tools
* https://www.researchgate.net/post/Can_anyone_suggest_an_image_labeling_tool_for_object_detection
* https://github.com/tzutalin/labelImg
```bash
git clone https://github.com/tzutalin/labelImg
```
* http://sloth.readthedocs.io/en/latest/
* https://github.com/yuyu2172/image-labelling-tool
* https://blog.playment.io/training-data-for-computer-vision/
* https://alpslabel.wordpress.com/
* https://github.com/commaai/commacoloring
* https://www.quora.com/What-is-the-best-image-labeling-tool-for-object-detection
* https://github.com/Labelbox/Labelbox/blob/master/LICENSE
* `git clone https://github.com/tzutalin/labelImg.git`
* https://oclavi.com/
* https://playment.io/image-annotation/
* https://blog.playment.io/training-data-for-computer-vision/

## Dataset / Data source / Datasource for ML / Deep Learning / Computer Vision Datasets
- https://projet.liris.cnrs.fr/voir/wiki/doku.php?id=datasets
- https://handong1587.github.io/computer_vision/2015/09/24/datasets.html
* **CIFAR-10**
  - One popular toy image classification dataset is the CIFAR-10 dataset. This dataset consists of 60,000 tiny images that are 32 pixels high and wide. Each image is labeled with one of 10 classes (for example “airplane, automobile, bird, etc”). These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images.
  - http://www.cs.toronto.edu/~kriz/cifar.html
* **Pima Indians**
* **Ionosphere**
  - http://cv-tricks.com/tensorflow-tutorial/understanding-alexnet-resnet-squeezenetand-running-on-tensorflow/
* Image segmentations
  - https://aws.amazon.com/public-datasets/spacenet/
  - http://www.cvlibs.net/datasets/kitti/eval_road.php

### Traffic Sign Datasets
* **German**
  - http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
* **LISA: Laboratory for Intelligent Safe Automobiles**
  - http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
* **Belgium**
  - https://btsd.ethz.ch/shareddata/
  - The images in this dataset are in an old .ppm format
  - Images are square-ish, but have different aspect ratios
  - The image quality is great, and there are a variety of angles and lighting conditions
  - The traffic signs occupy most of the area of each image, which allows to focus on object classification and not have to worry about finding the location of the traffic sign in the image (object detection).
  - Generally, neural network will take a fixed-size input, so some preprocessing is required.
  - Dataset considers all speed limit signs to be of the same class, regardless of the numbers on them. That’s fine, as long as we know about it beforehand and know what to expect.
  - Labels 26 and 27 are interesting to check
  - What are the sizes of the images? - The sizes seem to hover around 128x128.
  - This tells me that the image colors are the standard range of 0–255.
  * **Additional Notes**
    - There is one directory for each of the 62 classes (00000 - 00061)
    - Each directory contains the corresponding training images and one  text file with annotations, eg. GT-00000.csv [headers: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId]
    - In total are 4591 images for training.
    - On average for each physically distinct traffic sign there are 3 images available.
    - The images are PPM images (RGB color)
    - Names are:
      - XXXXX_YYYYY.ppm, XXXXX - pole number
      - running number for the views where the traffic sign is annotated. There is no temporal order of the images

### **Self Driving Car Datasets Semantic Segmentation**
- https://blog.playment.io/self-driving-car-datasets-semantic-segmentation/
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  - http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
* [KITTI - Karlsruhe Institute of Technology and Toyota Technological Institute](http://www.cvlibs.net/datasets/kitti/)
  - http://www.cvlibs.net/datasets/kitti/
  - http://www.cvlibs.net/datasets/kitti/eval_road.php
* [DUS - Daimler Urban Segmentation](http://www.6d-vision.com/scene-labeling)
  - http://www.6d-vision.com/scene-labeling
  - http://www.6d-vision.com/home
* [CityScapes](https://www.cityscapes-dataset.com/)
  - https://www.cityscapes-dataset.com/
  - https://www.cityscapes-dataset.com/benchmarks/
* [Mapillary Vista](https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html)
  - https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html
  - https://www.mapillary.com/dataset/vistas
* [Synthia](http://synthia-dataset.com/download-2/)
  - http://synthia-dataset.net/ 
* [Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets)
  - https://github.com/udacity/self-driving-car/tree/master/datasets


### 3D Datasets
- https://hackernoon.com/announcing-the-matterport3d-research-dataset-815cae932939
- https://niessner.github.io/Matterport/
- https://arxiv.org/pdf/1709.06158.pdf

### Others
- https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research
* Free datasets are available from places like Kaggle.com and UCI. 
  - https://www.kaggle.com/datasets
  - https://archive.ics.uci.edu/ml/datasets.html

## Datasets Detailed Review

1. [Cityscape](cityscape-dataset.md)
2. [Mapillary](mapillary-dataset.md)

## Labelling
- https://github.com/udacity/self-driving-car
- http://aid-driving.eu/active-learning-and-labeling/
- add situation-specific label
  *  for each image if it was day, rainy, if there were roadworks, close traffic participants (or far away), and many more things.
  * Image Properties
    - Roadworkds:
    - Cloudy
    - Traffic participants close
    - Traffic participants far
    - City
    - Rainy
    - Day
  * Transfer learning is the practice of taking an existing neural network trained on a specific task and retraining this neural network on another task. 
  * By using transfer learning we profit from existing lower level filters. By training on all classes at the same time the gradients from other classes influence the upper layers!

## Dataset Management
- https://autonomous-driving.org/2018/06/16/dataset-management-for-machine-learning/