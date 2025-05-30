# models/residual_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.res_block(x)
        down = self.pool(out)
        return out, down

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        up = self.upconv(x)
        if up.size() != skip_connection.size():
            up = F.interpolate(up, size=skip_connection.size()[2:], mode='bilinear', align_corners=True)
        out = torch.cat([up, skip_connection], dim=1)
        out = self.res_block(out)
        return out

class ResidualUNet(nn.Module):
    def __init__(self):
        super(ResidualUNet, self).__init__()
        self.encoder1 = DownSample(1, 64)
        self.encoder2 = DownSample(64, 128)
        self.encoder3 = DownSample(128, 256)
        self.encoder4 = DownSample(256, 512)

        self.middle = ResidualBlock(512, 1024)

        self.decoder4 = UpSample(1024, 512)
        self.decoder3 = UpSample(512, 256)
        self.decoder2 = UpSample(256, 128)
        self.decoder1 = UpSample(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1, down1 = self.encoder1(x)
        enc2, down2 = self.encoder2(down1)
        enc3, down3 = self.encoder3(down2)
        enc4, down4 = self.encoder4(down3)

        middle = self.middle(down4)

        up4 = self.decoder4(middle, enc4)
        up3 = self.decoder3(up4, enc3)
        up2 = self.decoder2(up3, enc2)
        up1 = self.decoder1(up2, enc1)

        out = self.final_conv(up1)
        return out
