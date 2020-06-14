import torch
import torch.nn as nn
import torch.nn.functional as F
# Build conv + Dilated conv
class NormalDilatedConv(nn.Module):
    '''
    Performs Normal 3x3 convolution and 5x5 dilated convolution. Concatenates both the output and downsizes the number of sum of channels to half using pointwise convolution.
    '''
    def __init__(self, in_channel, out_channel):
        super(NormalDilatedConv, self).__init__()
        self.normalconv = nn.Sequential(
                nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
                )

        self.dilatedconv = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channel)
        )

        self.Pointwise = nn.Conv2d(in_channels=(out_channel*2), out_channels=out_channel, kernel_size=1, padding=0, bias=False)


    def forward(self, x):
        out = self.normalconv(x)
        outdilated = self.dilatedconv(x)
        out = torch.cat([out, outdilated], dim=1)
        out = self.Pointwise(out)
        return out




class SuperRes(nn.Module):
    '''
    Increases channels by factor of 2 and performs pixel shuffle by a factor of 2. Thus doubling image dimension.
    '''
    def __init__(self, in_channel):
        super(SuperRes, self).__init__()
        self.Pointwise = nn.Conv2d(in_channels=in_channel, out_channels=(in_channel*2), kernel_size=1, padding=0, bias=False)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.Pointwise(x)
        x = self.pixelshuffle(x)
        return x


## DepthMaskNet

class DepthMaskNet(nn.Module):
    def __init__(self):
        super(DepthMaskNet, self).__init__()

        ##Encoder Block

        # bg and bg_fg (Will avoid using dilation convolution on the input itself)
        # self.bg = NormalDilatedConv(3, 32)
        # self.bg_fg = NormalDilatedConv(3, 32)

        self.bg = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
                )

        self.bg_fg = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
                )


        # Pointwise conv for bg and bg_fg merge.
        self.pointwise = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False)

        self.encode1 = NormalDilatedConv(32, 64)
        self.encode2 = NormalDilatedConv(64, 128)
        self.encode3 = NormalDilatedConv(128, 256)
        self.encode4 = NormalDilatedConv(256, 512)

        self.maxpool = nn.MaxPool2d(2, 2)

        ## Decoder block
        self.mask_decode1 = SuperRes(512)
        self.mask_decode2 = SuperRes(256)
        self.mask_decode3 = SuperRes(128)
        self.mask_decode4 = SuperRes(64)

        self.depth_decode1 = SuperRes(512)
        self.depth_decode2 = SuperRes(256)
        self.depth_decode3 = SuperRes(128)
        self.depth_decode4 = SuperRes(64)

        self.mask_outputConv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.depth_outputConv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False)


    def forward(self, f_bg, f_bg_fg):
        bg = self.bg(f_bg)
        fg = self.bg_fg(f_bg_fg)

        #Concatenate bg and bg_fg and reduce channel by half.
        x = torch.cat([bg, fg], dim=1)
        x = self.pointwise(x)

        #Encoding
        x = self.encode1(x)
        x = self.maxpool(x)
        x = self.encode2(x)
        x = self.maxpool(x)
        x = self.encode3(x)
        x = self.maxpool(x)
        x = self.encode4(x)
        x = self.maxpool(x)

        # Mask Decode
        mask = self.mask_decode1(x)
        mask = self.mask_decode2(mask)
        mask = self.mask_decode3(mask)
        mask = self.mask_decode4(mask)

        mask = self.mask_outputConv(mask)

        # Depth Decode
        depth = self.depth_decode1(x)
        depth = self.depth_decode2(depth)
        depth = self.depth_decode3(depth)
        depth = self.depth_decode4(depth)

        depth = self.depth_outputConv(depth)

        return mask, depth
