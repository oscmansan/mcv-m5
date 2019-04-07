# RetinaNet

Scientists at Facebook AI Research (FAIR) studied the reason why two-stage object detectors (like Faster R-CNN) outperforms one-stage detectors (like SSD) and they discovered that the foreground-background class imbalance during training was the central cause. In order to solve that, they trained a detector called RetinaNet which uses Focal loss, a variation of the cross entropy loss that addresses this class imbalance. Applying this, the learning is focused on the hard negatives examples.

RetinaNet is composed by a backbone network and two task-specific subnetworks.

First, images are processed by a Resnet, which output is introduced in a Feature Pyramid Network (FPN) in order to detect objects at different scales.

The detections of each pyramid level are feeded into a small Fully Connected Network (FCN), which predicts the probability of object presence at each spatial position for each of the anchors and object classes. The parameters of this subnet are shared between all across all pyramid levels.

In parallel, there is another small FCN attached to each pyramid level to get the offset from each anchor box to the nearest ground-truth object.

RetinaNet achieves state-of-the-art accuracy and speed, outperforming SSD, DSSD and R-FCN on both Accuracy and Speed vs. Accuracy Tradeoff and even improves the results of two-stage detectors like Faster R-CNN based on Inception-ResNet-V2-TDM.