# Week 2

## Goals

- [x] Run the provided code 
- [x] Train a network for other dataset
- [x] Implement a new Network
    - [x] Using an existing PyTorch implementation.
    - [x] Writing our own implementation.
- [x] Boost the performance of your networks 
- [x] Report+slides showing the achieved results
    - [x] Readme
    - [x] Report
    - [x] Slides

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

##### (a) Run the provided code

* The TsingHua-TenCent 100K (TT100K) dataset contains 64x64 crops of 221 different traffic signs. The training set contains 16527 images belonging to 45 classes. The validation set contains 1644 images belonging to 33 classes. The test set contains 8190 images belonging to 45 classes.
* A VGG-16 model has been trained on the TT100K dataset using the default configuration, fine-tuning from pre-trained weights on ImageNet. After 25 epochs, a 98.28% of accuracy is achieved in the training set, a 89.32% in the validation set and 95.94% in the test set.
* A VGG-16 model has been trained on the BelgiumTSC dataset, applying transfer learning from a the previous model trained on TT100K. The last fully-connected layer has been resized from 221 to 62 outputs, and trained from scratch. After only 12 epochs, a 98.45% of accuracy is achieved in the training set, a 97.26% in the validation set and 97.26% in the test set.