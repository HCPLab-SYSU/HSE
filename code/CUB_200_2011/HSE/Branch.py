'''
make the embeding layer using resnet
'''
import math

import torch
import torch.nn as nn
from model import Bottleneck

class Branch(nn.Module):
    def __init__(self, level='class'):
        self.inplanes = 1024 # ResNet params
        super(Branch, self).__init__()

        print "branch({}): (3 Bottlenecks) + fc".format(level)
        self.resnet50_layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet50_layer4(x)
            
        return x