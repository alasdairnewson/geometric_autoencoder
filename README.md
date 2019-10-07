## Processsing Simple Geometric Attributes with Autoencoders

This code implements the work in the following JMIV paper :

"Processsing Simple Geometric Attributes with Autoencoders", Alasdair Newson, Andrés Almansa, Yann Gousseau, Saïd Ladjal, Journal of Mathematical Imaging and Vision, 2019

### Abstract

Image synthesis is a core problem in modern deep learning, and many recent architectures such as autoencoders and Generative Adversarial networks produce spectacular results on highly complex data, such as images of faces or landscapes. While these results open up a wide range of new, advanced synthesis applications, there is also a severe lack of theoretical understanding of how these networks work. This results in a wide range of practical problems, such as difficulties in training, the tendency to sample images with little or no variability, and generalisation problems. In this paper, we propose to analyse the ability of the simplest generative network, the autoencoder, to encode and decode two simple geometric attributes : size and position. We believe that, in order to understand more complicated tasks, it is necessary to first understand how these networks process simple attributes. For the first property, we analyse the case of images of centred disks with variable radii. We explain how the autoencoder projects these images to and from a latent space of smallest possible dimension, a scalar. In particular, we describe both the encoding process and a closed-form solution to the decoding training problem in a network without biases, and show that during training, the network indeed finds this solution. We then investigate the best regularisation approaches which yield networks that generalise well. For the second property, position, we look at the encoding and decoding of Dirac delta functions, also known as "one-hot" vectors. We describe a hand-crafted filter that achieves encoding perfectly, and show that the network naturally finds this filter during training. We also show experimentally that the decoding can be achieved if the dataset is sampled in an appropriate manner. We hope that the insights given here will provide better understanding of the precise mechanisms used by generative networks, and will ultimately contribute to producing more robust and generalisable networks.


## Requirements
The code is written in Python 3.6 and requires the following packages :
* numpy
* scipy
* pickle
* imageio
* scikit-image
* matplotlib
* keras


## Setting up the code
* Install the required packages (with either pip or anaconda)
* Download the code from the GitHub repository :
```
bash git clone https://github.com/alasdairnewson/geometric_autoencoder
cd geometric_autoencoder
```

### Creating the databases

You will notice that there is no data folder. This is normal : in this work we have only concentrated on synthetic examples of geometric shapes. Therefore, a python script called "generate_data.py" is included which will generate certain datasets for you. For example, to create some train and test images of disks, type the following :

```
python generate_data.py --geometric_object disk
```

You can do this with the following predefined shapes :
* disks
* squares
* circles
* ellipses
* shifted disks

Note that you can use your own database as well, just modify the ```dataset_dir_train``` and  ```dataset_dir_test``` parameters in the ```geometric_autoencoder_keras.py``` file.