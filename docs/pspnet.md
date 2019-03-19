# PSPNet

Pyramid Scene Parsing Network (PSPNet) was the winner of the ImageNet scene parsing challenge 2016, the PASCAL VOC 2012 benchmark and the Cityscapes benchmark at that moment. By using a pyramid pooling module, PSPNet has surpassed state-of-the-art approaches such as FCN, DeepLab, and DilatedNet.

PSPNet authors realized that many errors in semantic segmentation and scene parsing are related to contextual relationship and global information for different receptive fields, such as mismatched relationships, confusion between categories and inconspicuous classes. Therefore, authors propose a global context prior representation that aims to solve these common issues and improve scene parsing.

PSPNet introduces the pyramid pooling module, a hierarchical global prior containing information with different scales and varying among different sub-regions. This module is placed upon the final layer feature maps of a CNN. Sub-region average pooling is performed for each feature map:
 
1. Global average pooling over each feature map, to generate a single bin output.
2. Divide the feature map into 2x2 sub-regions, then perform average pooling for each sub-region.
3. Divide the feature map into 3x3 sub-regions, then perform average pooling for each sub-region.
4. Divide the feature map into 6x6 sub-regions, then perform average pooling for each sub-region.

Then 1×1 convolution is performed for each pooled feature map to reduce the dimension of context representation to 1/N of the original one if the level size of the pyramid is N. Then, the low-dimension feature maps are upsampled to have the same size as the original feature maps via bilinear interpolation. Finally, different levels of features are concatenated as the final pyramid pooling global feature.

This pyramid parsing module is used to build the PSPNet architecture. In the first stage, a pretrained ResNet model with the dilated network strategy is used; the final feature map is 1/8 of the input image. Then, the pyramid pooling module is used to gather context information. This global prior is followed by a convolutional layer to generate the final prediction map. But at this point, the spatial dimension of the output volume is still 1/8 of the input image. Is it not lacking a final upsampling layer?