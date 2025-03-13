# Imports
from typing import Type, Any, Callable, Union, List, Optional

# PyTorch Imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import torchvision



# Class: OpenCVXRayNN
class OpenCVXRayNN(nn.Module):

    def __init__(self, channels, height, width, nr_classes):
        super(OpenCVXRayNN, self).__init__()
        

        # Add variables to class variables
        self.channels = channels
        self.height = height 
        self.width = width
        self.nr_classes = nr_classes

        # Change this architecture so it takes into account the previous variables
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



# Class: ChestXRayNN
class ChestXRayNN(nn.Module):
    def __init__(self, channels=3, height=64, width=64, nr_classes=3):
        super(ChestXRayNN, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * (height // 8) * (width // 8), 512)
        self.fc2 = nn.Linear(512, nr_classes)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Model: DenseNet121 for OpenCVXRayNN
# DenseNet121 para ChestXRayNN
class DenseNet121ChestXRayNN(torch.nn.Module):
    def __init__(self, channels=3, height=64, width=64, nr_classes=3):
        super(DenseNet121ChestXRayNN, self).__init__()

        # Backbone para extrair características
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        # Camadas FC
        # Calcula in_features
        _in_features = torch.rand(1, channels, height, width)
        _in_features = self.densenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Camada FC1 para classificação
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=512)
        self.fc2 = torch.nn.Linear(512, nr_classes)

    def forward(self, x):
        x = self.densenet121(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# DenseNet121 para OpenCVXRayNN
class DenseNet121OpenCVXRayNN(torch.nn.Module):
    def __init__(self, channels=3, height=64, width=64, nr_classes=2):
        super(DenseNet121OpenCVXRayNN, self).__init__()

        # Backbone para extrair características
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        # Camadas FC
        # Calcula in_features
        _in_features = torch.rand(1, channels, height, width)
        _in_features = self.densenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Camada FC1 para classificação
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=512)
        self.fc2 = torch.nn.Linear(512, nr_classes)

    def forward(self, x):
        x = self.densenet121(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Teste do modelo ChestXRayNN com DenseNet121
    model = DenseNet121ChestXRayNN(channels=3, height=64, width=64, nr_classes=3)
    print(model)

    # Testa o modelo com um Tensor aleatório
    test_input = torch.randn(1, 3, 64, 64)
    test_output = model(test_input)
    print(test_output.shape)

    # Teste do modelo OpenCVXRayNN com DenseNet121
    model = DenseNet121OpenCVXRayNN(channels=3, height=64, width=64, nr_classes=3)
    print(model)

    # Testa o modelo com um Tensor aleatório
    test_input = torch.randn(1, 3, 64, 64)
    test_output = model(test_input)
    print(test_output.shape)

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

    # Test the model with a random Tensor
    # Your code here
    test_input = torch.randn (1, channels, height, width)
    test_output = model (test_input)
    print (test_output.shape)


    # Define the model
    model = ChestXRayNN(
        channels=channels,
        height=height,
        width=width,
        nr_classes=nr_classes
    )
    print(model)

    # Test the model with a random Tensor
    # Your code here
    test_input = torch.randn (1, channels, height, width)
    test_output = model (test_input)
    print (test_output.shape)
