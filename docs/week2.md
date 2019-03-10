# Week 2 - Object Recognition

# Abstract

The goal of this week is to recognize the kind of object in a given picture, assigning a class to it. This is achieved using state-of-the-art convolutional neural networks for feature extraction and classification. The models, fine-tuned from ImageNet, have been trained on a set of provided datasets: [TT100k](https://cg.cs.tsinghua.edu.cn/traffic-sign/), [BelgiumTSC](https://btsd.ethz.ch/shareddata/) and [KITTI](http://www.cvlibs.net/datasets/kitti/). These datasets can be used to train a pedestrian, traffic sign or vehicle classifier among others. In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation. Moreover, a new architecture for image classification, OscarNet, has been implemented and evaluated.

## Completed tasks

- [x] (a) Run the provided code & transfer learning
- [x] (b) Train a network on another dataset
- [x] (c) Implement a new Network
    - [x] Using an existing PyTorch implementation.
    - [x] Writing our own implementation.
- [x] (d) Boost the performance of our networks 
- [x] (e) Report showing the achieved results
    - [x] README
    - [x] Slides
    - [x] Report

## Implementation

- Run the provided code:
	- VGG16 with TT100K dataset.
	- Transfer learning to BelgiumTSC dataset.
- Implement Networks:
    - Using an existing PyTorch implementation:
		- ResNet152
		- DenseNet161
		- Inception_v3
    - Writing our own implementation:
		- OscarNet
- Train each network with the following datasets:
	- BelgiumTSC
	- KITTI 
	- TT100K
- Boost the performance of the networks:
	- Tune meta-parameters
	- Data augmentation
	
## Results

### Standard

| Datasets | Networks | train  | val   | test  |
|----------|----------|--------|-------|-------|
| Belgium  | VGG-16   | 99,83  | 97,74 | -     |
|          | ResNet   | 100,00 | 98,89 | -     |
|          | OscarNet | 98,73  | 98,57 | -     |
| TT100k   | VGG-16   | 97,74  | 90,00 | 95,94 |
|          | ResNet   | 99,98  | 95,68 | 98,47 |
|          | OscarNet | 99,03  | 79,74 | 80,29 |
| KITTI    | VGG-16   | 99,90  | 98,09 | -     |
|          | ResNet   | 100,00 | 99,08 | -     |
|          | OscarNet | 99,85  | 98,06 | -     |

### With Data Augmentation and ReduceLROnPlateau

| Datasets | Networks |    | train | val   | test  |
|----------|----------|----|-------|-------|-------|
| Belgium  | VGG-16   | DA | 99,08 | 97,94 | -     |
|          | ResNet   | DA | 99,96 | 99,21 | -     |
| tt100k   | VGG-16   | DA | 98,49 | 91,35 | 97,36 |
|          | ResNet   | DA | 99,84 | 95,81 | 98,28 |
| KITTI    | ResNet   | DA | 99,99 | 99,31 | -     |
| Belgium  | VGG-16   | LR | 99,98 | 97,26 | -     |
|          | ResNet   | LR | 100,00| 98,93 | -     |
| tt100k   | VGG-16   | LR | 99,71 | 92,84 | 97,66 |
|          | ResNet   | LR | 99,95 | 95,68 | 98,35 |
