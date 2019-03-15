
# Week 3 - Image Semantic Segmentation

# Abstract

The goal of this week is to segment each object of the scene giving its associated label, which is known as Image Semantic Segmentation. This is achieved using state-of-the-art convolutional neural networks. The models, fine-tuned from CamVid, have been trained on a set of provided datasets: [CityScapes](https://www.cityscapes-dataset.com/), [Pascal2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) [synthia_rand_cityscapes](http://synthia-dataset.net/download-2/) and [KITTI](http://www.cvlibs.net/datasets/kitti/). In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation. Moreover, a new architecture has been implemented and evaluated.

## Completed tasks

- [ ] (a) Run the provided code
- [ ] (b) Read 2 papers in semantic segmentation networks
- [ ] (c) Implement a new Network
    - [ ] Using an existing PyTorch implementation.
    - [ ] Writing our own implementation.
- [ ] (d) Train a network for another dataset
- [ ] (e) Boost the performance of our networks 
- [ ] (f) Report showing the achieved results
    - [ ] README
    - [ ] Slides
    - [ ] Report

## Implementation

- Run the provided code:
	- FCN8 with CamVid dataset.
- Implement Networks:
    - Using an existing PyTorch implementation:
      - SegnetVGG
      - DeepLab
      - ResNetFCN
      - Tiramisu
    - Writing our own implementation
- Train each network with the following datasets:
	- Cityscapes
	- Pascal2012
  - Synthia_rand_cityscapes
	- KITTY
- Boost the performance of the networks:
	- Tune meta-parameters
	- Data augmentation
	
## Results

### Standard

| Datasets | Networks | train  | val   | test  |
|----------|----------|--------|-------|-------|


### With Data Augmentation and ReduceLROnPlateau

| Datasets | Networks |    | train | val   | test  |
|----------|----------|----|-------|-------|-------|
