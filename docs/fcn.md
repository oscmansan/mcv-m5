# FCN

In 2014, the FCN networks aimed to outperform state of the art networks by using a fully convolutional architecture
which takes an input of arbitrary size and produces an output of the correspondent size. 

State of the art classification models were adapted into fully convolutional networks and their learned representations
were transferred by fine tuning to the segmentation task.

The network used upsampling in order to reduce training and prediction times and to improve the consistency of the 
output.

The main feature of this network was that it allows you to treat the convolutional neural network as one giant filter. 
You can then spatially apply the neural net as a convolution to images larger than the original training image size, 
getting a spatially dense output.
