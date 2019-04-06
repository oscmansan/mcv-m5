# Faster R-CNN

Faster R-CNN was the basis of several 1st-place entries in ILSVRC and COCO 2015 different challenges like ImageNet localization or COCO detection. By using a Region Proposal Network (RPN), it outperformed previous architectures in both speed and accuracy. 

 The previous version, Fast R-CNN, achieves almost real-time rates excluding object proposal time, being this the bottleneck of the network. To solve it, the researchers designed a system with two modules: the Region Proposal Network (RPN), which predicts object bounds and objectness scores and the Fast R-CNN that uses the proposed regions as input.

  The main novelty of this architecture is that the RPN can share convolutional layers with the object detection network. They found out that convolutional feature maps used by region-based detectors can also be used for generating region proposals, so they can be retrieved with almost zero cost. 

RPNs returns a set of rectangular object proposals and an objectness score for each one. To generate them, for each position of a sliding window it tries several anchor boxes with different aspect ratios and scales, providing a wide range of results. Unlike other region proposal methods like MultiBox, RPNs are translation invariant, which means that a translated object will be equally located no matter it position in the image. This provides a wide range of scales and aspect ratios.

