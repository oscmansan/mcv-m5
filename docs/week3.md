
# Week 3 - Image Semantic Segmentation

# Abstract

The goal of this week is to segment all the different objects (e.g., person, vehicle, building, road, sky...) from the input image and recognize its category, which is known as Semantic Segmentation. This is achieved using state-of-the-art convolutional neural networks with an encoder-decoder structure. The different models, fine-tuned from classification on ImageNet and adapted to the segmentation task, have been trained on a set of provided datasets: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), [CityScapes](https://www.cityscapes-dataset.com/), [Pascal2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) [Synthia](http://synthia-dataset.net/download-2/) and [KITTI](http://www.cvlibs.net/datasets/kitti/). In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation. Moreover, a new Semantic Segmentation architecture has been integrated into the framework.

## Completed tasks

- [x] (a) Run the provided code
- [x] (b) Read 2 papers about semantic segmentation networks
- [x] (c) Implement a new network
- [x] (d) Train the network(s) on a different dataset
- [ ] (e) Boost the performance of the network(s)
- [ ] (f) Report showing the achieved results
    - [ ] README
    - [ ] Slides
    - [ ] Report

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

| dataset | network | train   | val     | test    |
|---------|---------|---------|---------|---------|
