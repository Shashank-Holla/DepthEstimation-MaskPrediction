import torch.nn as nn
import torch.nn.functional as F

class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()
        
        # Image size = 32. RF = 1
        #Convolution Layer-1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1), bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(3)
            

        )    

        self.pool1 = nn.MaxPool2d(2, 2)

        #Convolution Layer-2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(64)
            
        )       

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        #Convolution Block-5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


    def forward(self, x1):
        x2 = self.convblock1(x1)
        x3 = self.convblock1(x1+x2)
        x4 = self.pool1(x1+x2+x3)
        x5 = self.convblock1(x4)
        x6 = self.convblock1(x4+x5)
        x7 = self.convblock1(x4+x5+x6)
        x8 = self.pool1(x5+x6+x7)
        x9 = self.convblock1(x8)
        x10 = self.convblock1(x8+x9)
        x11 = self.convblock2(x8+x9+x10)
        x12 = self.gap(x11)
        x13 = self.convblock3(x12)
        x13 = x13.view(-1, 10)
        return x13