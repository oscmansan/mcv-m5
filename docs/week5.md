
# Week 5 - Object Detection

# Abstract

The goal of this week is to detect all the different objects (e.g., person, vehicle, building, road, sky...) from the input image and recognize its category, which is known as Object Detection. This is achieved using state-of-the-art convolutional neural networks and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) framework. The different models, fine-tuned from COCO dataset, have been trained on a set of provided datasets: Udacity, TT100K, and KITTI. In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation.

## Completed tasks

- [x] (a) Train an existing object detection network
- [x] (b) Read 2 papers about object detection networks
- [x] (c) Train the network(s) on a different dataset
- [x] (d) Boost the performance of the network(s)
- [x] (e) Report showing the achieved results
    - [x] README
    - [x] Slides
    - [x] Report

## Implementation

- Run the provided code:
	- RetinaNet
  - Mask R-CNN
- Train each network with the following datasets:
	- Udacity
	- TT100K
	- KITTI
- Boost the performance of the networks:
	- Hyperparameters tuning
	- Data augmentation
	
## Results

Training results:

| network    | dataset     | AP      | AP@50   | AP@75   |
|------------|-------------|---------|---------|---------|
| Mask R-CNN | COCO        | 0.370   | 0.586   | 0.403   |
|            | TT100K_val  | 0.587   | 0.837   | 0.670   |
|            | TT100K_test | 0.764   | 0.962   | 0.920   |
|            | Udacity_val | 0.209   | 0.408   | 0.188   |
|            | Udacity_test| 0.230   | 0.443   | 0.214   |
|            | KITTY_val   | 0.627   | 0.906   | 0.718   |
| RetinaNet  | COCO        | 0.367   | 0.559   | 0.393   |
|            | TT100K_val  | 0.524   | 0.777   | 0.583   |
|            | TT100K_test | 0.767   | 0.965   | 0.923   |
|            | Udacity_val | 0.227   | 0.442   | 0.201   |
|            | Udacity_test| 0.238   | 0.461   | 0.224   |
|            | KITTY_val   | 0.565   | 0.851   | 0.630   |

