# VGG

Karen Simonyan and Andrew Zisserman investigate the effect of the convolutional network depth on its accuracy in the 
large-scale image recognition setting. Rigorous evaluation of networks showed that prior-art configuration could be 
improved increasing the depth to 16-19 weight layers. 

VGG network is a deep convolutional neural network with very small 3x3 convolution filters, an architecture created by 
VGG (Visual Geometry Group, University of Oxford) for the ILSVRC-2014. Its architecture is based on the idea that 
convolutional neural networks need to be deep to represent visual data properly. 

The network strictly uses 3x3 filters with stride and pad of 1, along with 2x2 maxpooling layers with stride 2. All 
hidden layers are equipped with the ReLU activation function. There are two fully connected layers with 4096 units each, 
followed by a 1000 units (representing the ImageNet classes) fully connected layer and a soft-max classification layer. 

The combination of 2 3x3 convolutional layers has an effective receptive field of 5x5, by using 3 layers of 3×3 
filters, it has a 7×7 receptive field. This simulates larger filters while keeping the benefits of smaller filter 
sizes. One of the benefits is reducing the number of parameters, which fastens convergence and reduces overfitting 
problem. Also, with two convolution layers, two ReLU layers can be used instead of one.

As a result of the convolutional and pooling layers, the spatial size of the input volumes at each layer decreases, 
but the depth of the volumes increase due to the increased number of filters. Since the number of filters doubles 
after each maxpool layer, spatial dimensions are shrinked while depth grows.

To obtain the optimum deep learning layer structure, depth effect has been studied increasing the number of layers one
y one and observing their convergence and accuracy improvement until it slowed down. From VGG-11 to VGG-13 the 
additional conv helps the classification accuracy, VGG-16 still improved by adding number of layers, until VGG-19 
since the network did not further improve, authors stopped adding layers.

At the submission of ILSVRC 2014, VGGNet archieved 7.3% error rate only which obtained 1st runner up position at the
moment. Nevertheless, the network worked well on both image classification and localization tasks, using a form of 
localization as regression (pag. 10 of the paper). Also, VGG representations generalise well to other datasets, 
where they achieved state-of-the-art results.
