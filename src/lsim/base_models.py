from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LSiM_Base(torch.nn.Module):
    def __init__(self):
        super(LSiM_Base, self).__init__()
        self.channels = [32,96,192,128,128]
        self.featureMapSize = [55,26,12,12,12]

        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 32, 12, stride=4, padding=2),
            nn.ReLU(),
        )
        self.slice2 = torch.nn.Sequential(
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(32, 96, 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.slice3 = torch.nn.Sequential(
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(96, 192, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.slice4 = torch.nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.slice5 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.N_slices = 5
        self.layerList = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        outputs = namedtuple("LSiMBaseOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out



class LSiM_Skip(torch.nn.Module):
    def __init__(self):
        super(LSiM_Skip, self).__init__()
        self.channels = [32,64,128,128,64,64,32,32]
        self.featureMapSize = [55,26,12,12,12,26,55,55]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 12, stride=4, padding=2),
            nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.deconv3 = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.deconv2 = torch.nn.Sequential(
            nn.ConvTranspose2d(128+64, 64, 3, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
        )
        self.deconv1 = torch.nn.Sequential(
            nn.ConvTranspose2d(64+64, 32, 5, stride=2, padding=0),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.N_slices = 8
        self.layerList = [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv3, self.deconv2, self.deconv1, self.conv5]


    def forward(self, X):
        con1 = self.conv1(X)
        #print("con1: " + str(con1.shape))
        con2 = self.conv2(con1)
        #print("con2: " + str(con2.shape))
        con3 = self.conv3(con2)
        #print("con3: " + str(con3.shape))
        con4 = self.conv4(con3)
        #print("con4: " + str(con4.shape))

        dec3 = self.deconv3(con4)
        #print("dec3: " + str(dec3.shape) + "\t(con4)")
        dec2 = self.deconv2( torch.cat((con3, dec3), 1) )
        #print("dec2: " + str(dec2.shape) + "\t(con3+dec3)")
        dec1 = self.deconv1( torch.cat((con2, dec2), 1) )
        #print("dec1: " + str(dec1.shape) + "\t(con2+dec2)")
        con5 = self.conv5( torch.cat((con1, dec1), 1) )
        #print("con5: " + str(con5.shape) + "\t(con1+dec1)\n")

        outputs = namedtuple("LSiMSkipOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7', 'relu8'])
        out = outputs(con1, con2, con3, con4, dec3, dec2, dec1, con5)

        return out

#----------------------------------------------------------------------------------------------------------
# Code for Squeeze, Vgg, Alex and Resnet adapted from:
# Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018)
# at https://github.com/richzhang/PerceptualSimilarity
#----------------------------------------------------------------------------------------------------------

class Squeezenet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Squeezenet, self).__init__()
        squeeze_pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
        self.channels = [64,128,256,384,384,512,512]
        self.featureMapSize = [111,55,27,13,13,13,13]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), squeeze_pretrained_features[x])

        self.layerList = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5, self.slice6, self.slice7]


    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class Alexnet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Alexnet, self).__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.channels = [64,192,384,256,256]
        self.featureMapSize = [55,27,13,13,13]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])

        self.layerList = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class Vgg16(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.channels = [64,128,256,512,512]
        self.featureMapSize = [224,112,56,28,14]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        self.layerList = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


#----------------------------------------------------------------------------------------------------------
# Code for Deep-Flow-Prediction network (DfpNet) adapted from:
# Thuerey et al. Well, how accurate is it? A Study of Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations (2018)
# at https://github.com/thunil/Deep-Flow-Prediction
#----------------------------------------------------------------------------------------------------------

class DfpNet(nn.Module):
    def __init__(self, pretrained=True, channelExponent=6, dropout=0.):
        super(DfpNet, self).__init__()
        self.channels = [64,128,128,256,512,512,512]
        self.featureMapSize = [112,58,28,14,7,3,1]

        chn = int(2 ** channelExponent + 0.5)
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, chn, 4, 2, 1, bias=True))

        self.layer2 = self.blockUNet(chn  , chn*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2x= self.blockUNet(chn*2, chn*2, 'layer2x',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = self.blockUNet(chn*2, chn*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4 = self.blockUNet(chn*4, chn*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer5 = self.blockUNet(chn*8, chn*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6 = self.blockUNet(chn*8, chn*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
        self.N_slices = 7

        if pretrained:
            pretrained_dict = torch.load("../Models/Base/DfpNet_modelGa")
            temp = dict(pretrained_dict)
            for key in temp:
                if 'dlayer' in key:
                    pretrained_dict.pop(key)
            self.load_state_dict(pretrained_dict)
        else:
            self.apply(self.weights_init)

        self.layerList = [self.layer1, self.layer2, self.layer2x, self.layer3, self.layer4, self.layer5, self.layer6]

    def blockUNet(self, in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
        block = nn.Sequential()
        if relu:
            block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        if not transposed:
            block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
        else:
            block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2))
            block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
        if bn:
            block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
        if dropout>0.:
            block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
        return block
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        h = self.layer1(x)
        h_block1 = h
        h = self.layer2(h)
        h_block2 = h
        h = self.layer2x(h)
        h_block2x = h
        h = self.layer3(h)
        h_block3 = h
        h = self.layer4(h)
        h_block4 = h
        h = self.layer5(h)
        h_block5 = h
        h = self.layer6(h)
        h_block6 = h
        outputs = namedtuple("DfpNetOutputs", ['block1', 'block2', 'block2x', 'block3', 'block4', 'block5', 'block6'])
        out = outputs(h_block1, h_block2, h_block2x, h_block3, h_block4, h_block5, h_block6)

        return out
