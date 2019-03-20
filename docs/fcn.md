# FCN

The FCN networks aims to outperform current state of the art networks by using a fully convolutional architecture
which takes an input of arbitrary size and produces an output of the correspondent size.

State of the art classification models are adapted into fully convolutional networks and their learned representations
are transferred by fine tuning to the segmentation task.

The network uses upsampling in order to reduce training and prediction times and to improve the consistency of the 
output.

The main feature of this network is that it allows you to treat the convolutional neural network as one giant filter. 
You can then spatially apply the neural net as a convolution to images larger than the original training image size, 
getting a spatially dense output.
