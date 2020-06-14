import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, num_blocks, stride=1):
        super(BasicBlock, self).__init__()
        self.num_blocks = num_blocks
        self.convAndMaxpool = nn.Sequential(
        nn.Conv2d(in_channels = in_planes, out_channels = planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(planes),
        nn.ReLU()
        )

        if self.num_blocks != 0:
            self.residue = nn.Sequential(
                nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU()
                )
            self.shortcut = nn.Sequential()


    def forward(self, x):
        x = self.convAndMaxpool(x)
        if self.num_blocks != 0:
            R1 = self.residue(x)
            R1 += self.shortcut(x)
            x = x + R1
        return x

        
    
class MixedResNet(nn.Module):
    """
    Custom CNN class to prepare model with combination of (Convolution + MaxPool2d + batchNorm + Relu) and ResNet block (1 block).
    Resnet block are for alternate layers.
    num_blocks = 1 --> to add resnet block to (conv + Maxpool)
    num_blocks = 0 --> to have only (conv + Maxpool)

    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(MixedResNet, self).__init__()
        self.in_planes = 64
        
        # Preparation layer
        self.preparationlayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )       
        
        #MixedResnet layer
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, num_blocks, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
    
        out = self.preparationlayer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def MixedResNet9():
    """
    Custom CNN class to prepare model with combination of (Convolution + MaxPool2d + batchNorm + Relu) and ResNet block (1 block).
    Resnet block are for alternate layers.
    num_blocks = 1 --> to add resnet block to (conv + Maxpool)
    num_blocks = 0 --> to have only (conv + Maxpool)

    """
    print("MixedResnet model is now loaded.")
    print("Conv + Maxpool + BN + Relu every layer. Resnet blocks every alternate layer- 1st and 3rd layers.")
    return MixedResNet(BasicBlock, [1,0,1])