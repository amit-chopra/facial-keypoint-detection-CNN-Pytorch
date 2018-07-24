## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I





class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # input = (224,224)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # output size = (W/K) = 110
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # output size = (W-F)/S +1 = (110-5)/1 +1 = 106 
        # output size = (W/K) = 106/2 = 53
        
        self.conv2 = nn.Conv2d(32, 64, 5)
      
        self.pool = nn.MaxPool2d(2, 2)

            
        self.fc1 = nn.Linear(64*53*53, 50)
        
        # dropout with p=0.4
        #self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(50, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # input image = (224,224)
        # output size after 1st conv = (W-F)/S +1 = (224-5)/1 +1 = 220
        # self.pool = nn.MaxPool2d(2, 2)
        # output size after 1st pooling layer = 110
  
       # print(x.shape)
    
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        
        
        # output size after 2nd conv = (W-F)/S +1 = (110-5)/1 +1 = 106
        # output size after 1st pooling layer = 53
        
        x = self.pool(F.relu(self.conv2(x)))   
        print(x.shape)
                
        
        # flatten the inputs into a vector
       
        x = x.view(x.size(0),-1)
        
        # self.conv2 = nn.Conv2d(32, 64, 5)
                      
        # two linear layers
        x = F.relu(self.fc1(x))
        print(x.shape)
        
        
        x = F.relu(self.fc2(x))
        print(x.shape)
        print(x)
        
                       
        # a modified x, having gone through all the layers of your model, should be returned
        return x

