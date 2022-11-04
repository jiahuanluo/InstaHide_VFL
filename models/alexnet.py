# -*- coding: utf-8 -*-
# ---
# @File: alexnet.py
# @Author: Jiahuan Luo
# @Institution: Webank, Shenzhen, China
# @E-mail: luojiahuan001@gmail.com
# @Time: 2022/8/30
# ---
import torch
import torch.nn as nn

from torch.nn import Conv2d as ConvBlock

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet_Bottom(nn.Module):
    def __init__(self, model_config, num_classes=NUM_CLASSES):
        self.cfg = model_config
        super(AlexNet_Bottom, self).__init__()
        self.features = []
        self.layer0 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer1 = nn.Sequential(
            ConvBlock(64, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

        )
        # self.layer2 = nn.Sequential(
        #     ConvBlock(192, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        # )
        # self.layer3 = nn.Sequential(
        #     ConvBlock(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # self.layer4 = nn.Sequential(
        #     ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # if self.cfg.apply_passport:
        #     self.set_passport()
        for i in range(2):
            layer = getattr(self, f"layer{i}")
            self.features.append(layer)

    def forward(self, x):
        for feature_layer in self.features:
            x = feature_layer(x)
        # x = self.layer0(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = x.view(x.size(0), 256 * 4 * 2)
        return x

    def get_layer_output(self, x, layer='layer0'):
        for feature_layer in self.features:
            x = feature_layer(x)
            if getattr(self, layer) == feature_layer:
                return x


class Alexnet_Top(nn.Module):
    def __init__(self, model_config, num_classes=NUM_CLASSES):
        super(Alexnet_Top, self).__init__()
        # self.cfg = model_config
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

    def forward(self, x_a, x_b=None):
        if x_b is not None:
            x = torch.cat([x_a, x_b], dim=-1)
        else:
            x = x_a
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x
