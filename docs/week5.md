
# Week 5 - Object Detection

# Abstract

The goal of this week is to detect objects (e.g. car, truck, pedestrian, cyclist, traffic sign) in the images, specifying their position with a bounding box, and recognize their category. This is known as object detection. For this task, we used the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) framework, open sourced by Facebook Research. The different models, fine-tuned from ImageNet, have been trained on a set of provided datasets: TT100K, Udacity, and KITTI. In addition, the performance of the models has been boosted by tuning the hyperparameters and performing data augmentation. The performance of the trained models has been measured using Average Precision (AP) and Average Recall (AR).

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

- Run the provided code and train the following networks on COCO:
    - Mask R-CNN
    - RetinaNet
- Train each network with the following datasets:
    - TT100K
    - Udacity
    - KITTI
- Boost the performance of the networks:
    - Hyperparameters tuning
    - Data augmentation
	
## Results

Training results:

| network    | dataset           | AP    | AP@50 | AP@75 | AR@1  | AR@10 | AR@100 |
|------------|-------------------|-------|-------|-------|-------|-------|--------|
| Mask R-CNN | coco_2014_minival | 0.370 | 0.586 | 0.403 | 0.306 | 0.477 | 0.500  |
| Mask R-CNN | tt100k_valid      | 0.587 | 0.837 | 0.670 | 0.595 | 0.610 | 0.610  |
| Mask R-CNN | tt100k_test       | 0.764 | 0.962 | 0.920 | 0.791 | 0.812 | 0.812  |
| Mask R-CNN | udacity valid     | 0.209 | 0.408 | 0.188 | 0.169 | 0.306 | 0.308  |
| Mask R-CNN | udacity_test      | 0.230 | 0.443 | 0.214 | 0.179 | 0.323 | 0.325  |
| Mask R-CNN | kitti_valid       | 0.627 | 0.906 | 0.718 | 0.331 | 0.672 | 0.679  |
| RetinaNet  | coco_2014_minival | 0.367 | 0.559 | 0.393 | 0.320 | 0.503 | 0.528  |
| RetinaNet  | tt100k_valid      | 0.524 | 0.777 | 0.583 | 0.600 | 0.625 | 0.625  |
| RetinaNet  | tt100k_test       | 0.767 | 0.965 | 0.923 | 0.798 | 0.820 | 0.820  |
| RetinaNet  | udacity valid     | 0.227 | 0.442 | 0.201 | 0.187 | 0.342 | 0.348  |
| RetinaNet  | udacity_test      | 0.238 | 0.461 | 0.224 | 0.190 | 0.353 | 0.363  |
| RetinaNet  | kitti_valid       | 0.565 | 0.851 | 0.630 | 0.298 | 0.615 | 0.628  |
