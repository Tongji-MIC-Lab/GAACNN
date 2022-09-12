import torch
import torch.nn as nn
import math


class MeanShift(nn.Conv2d):
    def __init__(self,
                 rgb_range=1.0,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self,
                 scale,
                 channels,
                 bn=False,
                 act=False,
                 bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(channels, 4 * channels, 3, 1, 1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(channels))
                if act:
                    m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(channels, 9 * channels, 3, 1, 1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(channels))

            if act:
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError
        super().__init__(*m)


class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACBlock, self).__init__()
        self.conv1x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 3), 1, (0, 1)))
        self.conv3x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0)))
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1)))

    def forward(self, x):
        conv3x1 = self.conv3x1(x)
        conv1x3 = self.conv1x3(x)
        conv3x3 = self.conv3x3(x)
        return conv3x1 + conv1x3 + conv3x3


class ACNet(nn.Module):
    def __init__(self,
                 # scale=2,
                 in_channels=3,
                 out_channels=3,
                 num_features=64,
                 num_blocks=17,
                 rgb_range=1.0):
        super(ACNet, self).__init__()
        # self.scale = scale
        self.num_blocks = num_blocks
        self.num_features = num_features

        # pre and post process
        self.sub_mean = MeanShift(rgb_range=rgb_range, sign=-1)
        self.add_mena = MeanShift(rgb_range=rgb_range, sign=1)

        # AB module
        self.blk1 = ACBlock(in_channels, num_features)
        for idx in range(1, num_blocks):
            self.__setattr__(f"blk{idx + 1}", nn.Sequential(nn.ReLU(inplace=True), ACBlock(num_features, num_features)))

        # MEB
        self.lff = nn.Sequential(
            nn.ReLU(inplace=False),
            # Upsampler(scale, num_features),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1)
        )
        self.hff = nn.Sequential(
            nn.ReLU(inplace=False),
            # Upsampler(scale, num_features),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1)
        )

        # HFFEB
        self.fusion = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        inputs = self.sub_mean(x)
        blk1 = self.blk1(inputs)

        high = blk1
        tmp = blk1
        for idx in range(1, self.num_blocks):
            tmp = self.__getattr__(f"blk{idx + 1}")(tmp)
            high = high + tmp

        lff = self.lff(blk1)
        hff = self.hff(high)

        fusion = self.fusion(lff + hff)
        output = self.add_mena(fusion)
        return output