
# Week 3 - Image Semantic Segmentation

# Abstract

The goal of this week is to segment all the different objects (e.g., person, vehicle, building, road, sky...) from the input image and recognize its category, which is known as Semantic Segmentation. This is achieved using state-of-the-art convolutional neural networks with an encoder-decoder structure. The different models, fine-tuned from classification on ImageNet and adapted to the segmentation task, have been trained on a set of provided datasets: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), [CityScapes](https://www.cityscapes-dataset.com/), [Pascal2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) [Synthia](http://synthia-dataset.net/download-2/) and [KITTI](http://www.cvlibs.net/datasets/kitti/). In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation. Moreover, a new Semantic Segmentation architecture has been integrated into the framework.

## Completed tasks

- [x] (a) Run the provided code
- [x] (b) Read 2 papers about semantic segmentation networks
- [x] (c) Implement a new network
- [x] (d) Train the network(s) on a different dataset
- [x] (e) Boost the performance of the network(s)
- [x] (f) Report showing the achieved results
    - [x] README
    - [x] Slides
    - [x] Report

## Implementation

- Run the provided code:
	- FCN8 on CamVid
- Implement a new network:
	- PSPNet
- Train each network with the following datasets:
	- CamVid
	- Cityscapes
	- KITTI
- Boost the performance of the networks:
	- Hyperparameters tuning
	- Data augmentation
	
## Results

Training results:

| dataset    | network   | train   | val     | test    |
|------------|-----------|---------|---------|---------|
| KITTI      | FCN8      | 0.721   | 0.465   | -       |
|            | PSPNet50  | 0.697   | 0.470   | -       |
|            | PSPNet101 | 0.708   | 0.486   | -       |
| CamVid     | FCN8      | 0.706   | 0.654   | 0.552   |
|            | PSPNet50  | 0.552   | 0.705   | 0.462   |
|            | PSPNet101 | 0.709   | 0.679   | 0.595   |
| Cityscapes | FCN8      | 0.584   | 0.557   | -       |

Transfer learning results:

| dataset        | network   | train   | val     | test    |
|----------------|-----------|---------|---------|---------|
| CamVid → KITTI | FCN8      | 0.537   | 0.436   | -       |
| VOC → KITTI    | FCN8      | 0.721   | 0.465   | -       |