import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, channels_in=128, channels_out=128, padding_mode="circular", use_norm=False, channels_per_group=1):
        """
        If channels_per_group=1, we have GroupNorm = InstanceNorm2d. 
        For the Autoencoder, channels_per_group > 1 does not give good results.
        """
        super(DownBlock, self).__init__()
        if use_norm:
            self.conv = nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, 3,
                            stride=2, padding=1, padding_mode=padding_mode),
                    nn.GroupNorm(num_groups=channels_out // channels_per_group, num_channels=channels_out, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channels_out, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                    nn.GroupNorm(num_groups=channels_out // channels_per_group, num_channels=channels_out, affine=True),
                    nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, 3,
                            stride=2, padding=1, padding_mode=padding_mode),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channels_out, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                    nn.LeakyReLU(0.2, inplace=True))            

    def forward(self, x):
        return self.conv(x)


class InBlock(nn.Module):
    def __init__(self, channels_in=1, channels_out=128, padding_mode="circular", use_norm=False,  channels_per_group=1):
        """
        If channels_per_group=1, we have GroupNorm = InstanceNorm2d. 
        For the Autoencoder, channels_per_group > 1 does not give good results.
        """
        super(InBlock, self).__init__()
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                nn.GroupNorm(num_groups=channels_out // channels_per_group, num_channels=channels_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                nn.LeakyReLU(0.2, inplace=True))           

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, channels_in=128, channels_out=128, padding_mode="circular", use_norm=False, channels_per_group=1):
        """
        If channels_per_group=1, we have GroupNorm = InstanceNorm2d. 
        For the Autoencoder, channels_per_group > 1 does not give good results.
        """
        super(UpBlock, self).__init__()
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride=1,
                            padding=1, padding_mode=padding_mode),
                nn.GroupNorm(num_groups=channels_out // channels_per_group, num_channels=channels_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels_out, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                nn.GroupNorm(num_groups=channels_out // channels_per_group, num_channels=channels_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride=1,
                            padding=1, padding_mode=padding_mode),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels_out, channels_out, 3,
                            stride=1, padding=1, padding_mode=padding_mode),
                nn.LeakyReLU(0.2, inplace=True))
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear',
        #                      align_corners=True)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, channels=128, padding_mode="circular", use_norm=False, use_sigmoid=False):
        super(Autoencoder, self).__init__()
        self.channels = channels 
        self.padding_mode = padding_mode
        self.use_norm = use_norm 
        self.use_sigmoid = use_sigmoid

        # Encoder
        self.conv1 = InBlock(1, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm)
        #nn.Conv2d(1, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.conv2 = DownBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm)
        #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.conv3 = DownBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm)
        #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.conv4 = DownBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm)
        #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.conv5 = DownBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm)

        self.conv5_b = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)

        # Decoder
        self.upconv1 = UpBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm) #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.upconv2 = UpBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm) #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode)
        self.upconv3 = UpBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm) #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode) 
        self.upconv4 = UpBlock(self.channels, self.channels, padding_mode=self.padding_mode, use_norm=self.use_norm) #nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode=self.padding_mode) 
        self.final_conv = nn.Conv2d(self.channels, 1, kernel_size=1, padding_mode=self.padding_mode)
        
        self.sigmoid = nn.Sigmoid() 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: input image (batch_size, 1, H, W)
        
        """
        
        # Encoder
        x1 = self.conv1(x) # H x W, 128x128
        x2 = self.conv2(x1) # H/2 x W/2, 64x64 
        x3 = self.conv3(x2) # H/4 x W/4, 32x32 
        x4 = self.conv4(x3) # H/8 x W/8, 16x16
        x5 = self.conv5(x4) # H/8 x W/8, 16x16
        # Bottleneck
        x_b = self.conv5_b(x5)

        # Decoder
        x_up1 = self.upconv1(x_b,target_size=x4.shape[-2:]) # H/4 x W/4, 32x32 
        x_up2 = self.upconv2(x_up1,target_size=x3.shape[-2:]) # H/2 x W/2, 64x64
        x_up3 = self.upconv3(x_up2,target_size=x2.shape[-2:]) # H x W, 128x128
        x_up4 = self.upconv4(x_up3,target_size=x1.shape[-2:]) # H x W, 128x128
        out = self.final_conv(x_up4) 
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out

    """
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))  # convd1 + relu1
        x1_down = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)  # down1

        x2 = self.relu(self.conv2(x1_down))  # convd2 + relu2
        x2_down = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)  # down2

        x3 = self.relu(self.conv3(x2_down))  # convd3 + relu3
        x3_down = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)  # down3

        x4 = self.relu(self.conv4(x3_down))  # convd4 + relu4
        x4 = self.relu(self.conv4_b(x4))     # conv4

        # Decoder
        x_up1 = F.interpolate(x4, scale_factor=2.0, mode='bilinear', align_corners=False)  # up1
        x_up1 = self.relu(self.upconv1(x_up1)) 

        x_up2 = F.interpolate(x_up1, scale_factor=2.0, mode='bilinear', align_corners=False)  # up2
        x_up2 = self.relu(self.upconv2(x_up2))  

        x_up3 = F.interpolate(x_up2, scale_factor=2.0, mode='bilinear', align_corners=False)  # up3
        x_up3 = self.relu(self.upconv3(x_up3))  

        out = self.final_conv(x_up3)  # convu4
        return out
    """