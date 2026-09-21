#!/usr/bin/env python3
"""
判别器模型
"""

import torch
import torch.nn as nn


class SimpleUNetDiscriminator(nn.Module):
    """U-Net判别器，输出逐像素、未经过 sigmoid 的真/假 logits。"""

    def __init__(self, in_channels=3):
        super().__init__()

        self.enc1 = self._block(in_channels, 64, batch_norm=False)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.mid = self._block(512, 512)

        self.dec4 = self._block(1024, 256)
        self.dec3 = self._block(512, 128)
        self.dec2 = self._block(256, 64)
        self.dec1 = self._block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, out_channels, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        mid = self.mid(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upsample(mid), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

        return self.final_conv(d1)
