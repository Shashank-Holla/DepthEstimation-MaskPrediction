import torch
import torch.nn as nn
import torch.nn.functional as F


dropout_value = 0.3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Image size = 32. RF = 1
        #Convolution Layer-1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #Output size- 32. RF=3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #Output size- 32. RF=5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #Output size- 32. RF=5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #Output size- 32. RF=5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16 RF = 6 Jout = 2

        #Convolution Layer-2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Output size- 16. RF=10
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Output size- 16. RF=14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) 

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8 RF = 22 Jout = 4

        #Convolution Layer-3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #Output size- 8. RF=22
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), #Output size- 8. RF=30
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )

        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 4 RF = 50 Jout = 8

        #Convolution Layer-4 294K parameters
        #Depthwise Seperable Convolution
        self.DepthwiseSepConv = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), 
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            # nn.Dropout(dropout_value)
            
            #Depthwise Convolution #Output size- 4. RF=46
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, groups = 128, bias=False),
            #Pointwise Convolution
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, groups = 256, bias=False),
            #Pointwise Convolution
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_value)
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        #Convolution Block-5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.DepthwiseSepConv(x)
        x = self.gap(x)
        x = self.convblock5(x)
        #print("Dimension of x",x.shape)
        x = x.view(-1, 10)
        return x
