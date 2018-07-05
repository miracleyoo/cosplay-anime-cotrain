# coding: utf-8
# Author: Miracle Yoo

import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class CoalesceNet(nn.Module):

    def __init__(self, inplanes, planes, opt, stride=1):
        super(CoalesceNet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(planes*8*8, opt.NUM_CLASSES)
        self.planes = planes

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out.view(-1, self.planes * 8 * 8)
        out = self.fc(out)

        return out


class Net(nn.Module):
    def __init__(self, model, coalesce_net):
        super(Net, self).__init__()
        # 取掉model的后两层
        self.resnet_layer_cos = nn.Sequential(*list(model.children())[:-2])
        self.resnet_layer_ani = nn.Sequential(*list(model.children())[:-2])
        self.coalesce = coalesce_net

    def forward(self, x1, x2):
        out_cos = self.resnet_layer_cos(x1)
        if self.is_train:
            out_ani = self.resnet_layer_ani(x2)
            x = [out_cos, out_ani]
            x = torch.cat(x, dim=1)
        else:
            x = out_cos
        x = self.coalesce(x)
        return x


def train_dou_resnet(resnet, opt):
    return Net(resnet, CoalesceNet(4096, 1024, opt))
