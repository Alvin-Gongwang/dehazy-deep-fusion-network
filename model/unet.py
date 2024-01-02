# /dehazyDeepFusionNetwork/model/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2, stride=2)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1],dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_conv1 = DoubleConv(3, 64)
        self.double_conv2 = DoubleConv(64, 128)
        self.double_conv3 = DoubleConv(128, 256)
        self.double_conv4 = DoubleConv(256, 512)
        self.double_conv5 = DoubleConv(512, 1024)
        self.double_conv6 = DoubleConv(1024, 512)
        self.double_conv7 = DoubleConv(512, 256)
        self.double_conv8 = DoubleConv(256, 128)
        self.double_conv9 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64,3, kernel_size=3, stride=1, padding=1)
        self.down = nn.MaxPool2d(2, stride=2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        x1 = self.double_conv1(x)
        x = self.down(x1)
        x2 = self.double_conv2(x)
        x = self.down(x2)
        x3 = self.double_conv3(x)
        x = self.down(x3)
        x4 = self.double_conv4(x)
        x = self.down(x4)
        x = self.double_conv5(x)
        x = self.up4(x)
        x = torch.cat([x4, x], dim=1)
        x = self.double_conv6(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.double_conv7(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.double_conv8(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.double_conv9(x)
        x = self.out_conv(x)
        return x

if __name__ == '__main__':

    # x = torch.randn(1, 3, 1088, 1920)
    x = torch.randn(1, 3, 224, 224)
    model = UNet()
    # y = model(x)
    # print(y.shape)
    # summary(model, input_size=[(1, 3, 224, 224)], batch_size=1, device="cpu")
    writer = SummaryWriter(log_dir=r'../log')
    writer.add_graph(model, input_to_model=x, verbose=False)
    writer.close()
