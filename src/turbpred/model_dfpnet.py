import torch
import torch.nn as nn
import torch.nn.functional as F

from turbpred.model_diffusion_blocks import SinusoidalPositionEmbeddings


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, stride=2, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=stride, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block


class DfpNet(nn.Module):
    def __init__(self, inChannels=3, outChannels=3, blockChannels=6, dropout=0.):
        super(DfpNet, self).__init__()
        c = blockChannels

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(inChannels, c, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(c  , c*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, size=3, stride=1, pad=1)
        self.layer2b= blockUNet(c*2, c*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(c*2, c*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(c*4, c*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer5 = blockUNet(c*8, c*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, size=3, pad=1)
        self.layer6 = blockUNet(c*8, c*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=3, pad=1)
    
        self.dlayer6 = blockUNet(c*8, c*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer5 = blockUNet(c*16,c*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer4 = blockUNet(c*16,c*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer3 = blockUNet(c*8, c*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(c*4, c*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(c*4, c  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout, stride=1 )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(c*2, outChannels, 4, 2, 1, bias=True))

    def forward(self, x, unused=None):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1



class BlockUNetTimeEmb(nn.Module):
    def __init__(self, in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, stride=2, pad=1, dropout=0., time_dim=-1):
        super().__init__()

        block = nn.Sequential()
        if relu:
            block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        if not transposed:
            block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
        else:
            block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=stride, mode='bilinear')) # Note: old default was nearest neighbor
            # reduce kernel size by one for the upsampling (ie decoder part)
            block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
        if bn:
            block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
        if dropout>0.:
            block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

        self.block = block

        self.mlp_time = nn.Sequential(nn.GELU(), nn.Linear(time_dim, out_c))

    def forward(self, x, time_emb):
        cond = self.mlp_time(time_emb).unsqueeze(2).unsqueeze(3)

        h = self.block(x)
        return h + cond




class DfpNetTimeEmbedding(nn.Module):
    def __init__(self, inChannels=3, outChannels=3, blockChannels=6, dropout=0.):
        super(DfpNetTimeEmbedding, self).__init__()
        c = blockChannels

        dim = 128
        time_dim = dim * 4
        self.input_time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(inChannels, c, 4, 2, 1, bias=True))

        self.layer2 = BlockUNetTimeEmb(c  , c*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, time_dim=time_dim, size=3, stride=1, pad=1)
        self.layer2b= BlockUNetTimeEmb(c*2, c*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, time_dim=time_dim, )
        self.layer3 = BlockUNetTimeEmb(c*2, c*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, time_dim=time_dim, )
        self.layer4 = BlockUNetTimeEmb(c*4, c*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout, time_dim=time_dim, )
        self.layer5 = BlockUNetTimeEmb(c*8, c*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, time_dim=time_dim, size=3, pad=1)
        self.layer6 = BlockUNetTimeEmb(c*8, c*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, time_dim=time_dim, size=3, pad=1)
    
        self.dlayer6 = BlockUNetTimeEmb(c*8, c*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,)
        self.dlayer5 = BlockUNetTimeEmb(c*16,c*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,)
        self.dlayer4 = BlockUNetTimeEmb(c*16,c*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,) 
        self.dlayer3 = BlockUNetTimeEmb(c*8, c*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,)
        self.dlayer2b= BlockUNetTimeEmb(c*4, c*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,)
        self.dlayer2 = BlockUNetTimeEmb(c*4, c  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout, time_dim=time_dim,stride=1 )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(c*2, outChannels, 4, 2, 1, bias=True))

    def forward(self, x, time):
        out1 = self.layer1(x)

        t = self.input_time_mlp(time)

        out2 = self.layer2(out1, t)
        out2b= self.layer2b(out2, t)
        out3 = self.layer3(out2b, t)
        out4 = self.layer4(out3, t)
        out5 = self.layer5(out4, t)
        out6 = self.layer6(out5, t)
        dout6 = self.dlayer6(out6, t)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5, t)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4, t)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3, t)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b, t)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2, t)
        dout2_out1 = torch.cat([dout2, out1], 1)

        dout1 = self.dlayer1(dout2_out1)
        return dout1