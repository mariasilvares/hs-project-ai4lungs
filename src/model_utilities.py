# Imports
from typing import Type, Any, Callable, Union, List, Optional

# PyTorch Imports
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision



import torch
import torch.nn as nn
import torch.nn.functional as F



# Class: 
class OpenCVXRayNN(nn.Module):

    def __init__(self, channels, height, width, nr_classes):
        super(OpenCVXRayNN, self).__init__()
        

        # Add variables to class variables
        self.channels = channels
        self.height = height 
        self.width = width
        self.nr_classes = nr_classes

        # TODO: Change this architecture so it takes into account the previous variables
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(channels, 6, 5) # Toma channels como entrada em vez de ser fixo em 1
        self.conv2 = nn.Conv2d(6, 16, 5)


        # Calculo do tamanho da saída 
        # Primeira convolução
        conv1_height = (height-5) + 1
        conv1_width = (width-5) + 1

        # Primeira camada de Pooling
        pool1_height = conv1_height // 2
        pool1_width = conv1_width // 2

        # Segunda convolução
        conv2_height = (pool1_height-5) + 1
        conv2_width = (pool1_width-5) + 1

        # Segunda camada de Pooling
        pool2_height = conv2_height // 2
        pool2_width = conv2_width // 2

        # Número total
        conv_out_size = 16*pool2_height*pool2_width 



        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(conv_out_size, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nr_classes) # Saída agora é nr_classes em vez do valor fixo 10

        return


    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output



if __name__ == "__main__":

    # Create the dimensions of the data (hint: might be useful to run the dataset_utilities.py again!)
    channels = 3 # Add here
    height = 64 # Add here
    width = 64  # Add here
    nr_classes = 3 # Add here

    # Define the model
    model = OpenCVXRayNN(
        channels=channels,
        height=height,
        width=width,
        nr_classes=nr_classes
    )
    print(model)

    # TODO: Test the model with a random Tensor
    # Your code here
    test_input = torch.randn (1, channels, height, width)
    test_output = model (test_input)
    print (test_output.shape)