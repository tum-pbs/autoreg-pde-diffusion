import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lsim.base_models import *
from lsim.dataset_distance import *


class DistanceModel(nn.Module):
    def __init__(self, baseType="lsim", initBase="pretrained", initLin=0.25, featureDistance="L2",
                 frozenLayers=[], normMode="normDist", useNormUpdate=False, isTrain=False, useGPU=False):
        super(DistanceModel, self).__init__()
        self.useGPU = useGPU
        self.useDropout = True
        self.featureDistance = featureDistance
        self.normMode = normMode
        self.useNormUpdate = useNormUpdate
        self.isTrain = isTrain

        # create base model and feature map weights (lin)
        if baseType == "alex":
            self.basenet = Alexnet(pretrained=True)
        elif baseType == "vgg":
            self.basenet = Vgg16(pretrained=True)
        elif baseType == "squeeze":
            self.basenet = Squeezenet(pretrained=True)
        elif baseType == "dfp":
            self.basenet = DfpNet(pretrained=True)
        elif baseType == "lsim":
            self.basenet = LSiM_Base()
        elif baseType == "lsimSkip":
            self.basenet = LSiM_Skip()
        else:
            raise ValueError('Unknown base network type.')

        self.normAcc = []
        self.normM2 = []
        for i in range(self.basenet.N_slices):
            if self.useGPU:
                self.normAcc += [torch.tensor([0.0], requires_grad=False).cuda()]
                self.normM2  += [torch.tensor([0.0], requires_grad=False).cuda()]
            else:
                self.normAcc += [torch.tensor([0.0], requires_grad=False)]
                self.normM2  += [torch.tensor([0.0], requires_grad=False)]
        self.normCount = [0] * self.basenet.N_slices

        self.lin0 = self.linearLayer(self.basenet.channels[0], self.basenet.featureMapSize[0])
        self.lin1 = self.linearLayer(self.basenet.channels[1], self.basenet.featureMapSize[1])
        self.lin2 = self.linearLayer(self.basenet.channels[2], self.basenet.featureMapSize[2])
        self.lin3 = self.linearLayer(self.basenet.channels[3], self.basenet.featureMapSize[3])
        self.lin4 = self.linearLayer(self.basenet.channels[4], self.basenet.featureMapSize[4])
        self.linear = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if baseType in ["squeeze", "dfp", "lsimSkip"]:
            self.lin5 = self.linearLayer(self.basenet.channels[5], self.basenet.featureMapSize[5])
            self.lin6 = self.linearLayer(self.basenet.channels[6], self.basenet.featureMapSize[6])
            self.linear = self.linear + [self.lin5, self.lin6]
            if baseType == "lsimSkip":
                self.lin7 = self.linearLayer(self.basenet.channels[7], self.basenet.featureMapSize[7])
                self.linear = self.linear + [self.lin7]

        # override pretrained initialization if required
        if initBase == "pretrained":
            pass
        elif initBase == "randomSmall":
            self.apply(self.initializeRandomSmall)
        elif initBase == "randomLarge":
            self.apply(self.initializeRandomLarge)
        elif initBase == "xavier":
            self.apply(self.initializeXavier)
        elif initBase == "layerwiseMean":
            self.apply(self.initializeLayerwiseMean)
        else:
            raise ValueError("Unknown initialization.")

        # initialize feature map weights
        for linLayer in self.linear:
            for layer in linLayer:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data.fill_(initLin)

        # freeze layers of basenet
        for i in frozenLayers:
            if i < len(self.basenet.layerList):
                for param in self.basenet.layerList[i].parameters():
                    param.requires_grad = False

        # GPU and evaluation mode setup
        if self.useGPU:
            self.cuda()

        if self.isTrain:
            self.train()
        else:
            self.eval()


    def forward(self, x):
        input1 = x["reference"]
        input2 = x["other"]

        if self.useGPU:
            input1 = input1.cuda()
            input2 = input2.cuda()

        sizeIn = input1.shape
        input1 = input1.view(sizeIn[0]*sizeIn[1], sizeIn[2], sizeIn[3], sizeIn[4])
        input2 = input2.view(sizeIn[0]*sizeIn[1], sizeIn[2], sizeIn[3], sizeIn[4])

        self.clampWeights()

        outBase1 = self.basenet(input1)
        outBase2 = self.basenet(input2)

        result = torch.tensor([[0.0]]).cuda() if self.useGPU else torch.tensor([[0.0]])

        for i in range( len(outBase1) ):
            updateNorm = self.isTrain and self.useNormUpdate
            normalized1 = self.normalizeTensor(outBase1[i], i, updateAcc=updateNorm)
            normalized2 = self.normalizeTensor(outBase2[i], i, updateAcc=updateNorm)

            if self.featureDistance == "L1":
                diff = torch.abs(normalized2 - normalized1)
            elif self.featureDistance == "L2":
                diff = (normalized2 - normalized1)**2
            else:
                raise ValueError('Unknown feature distance.')

            weightedDiff = self.linear[i](diff)
            result = result + torch.mean( torch.mean(weightedDiff, dim=3), dim=2 )
            del weightedDiff

        if self.featureDistance == "L2":
            result = torch.sqrt(result)

        return result.view(sizeIn[0], sizeIn[1])


    # input two 4D/3D numpy arrays in order [batch, width, height, channels] or
    # [width, height, channels] and return a distance of shape [batch] or [1]
    def computeDistance(self, input1, input2, interpolateTo=224, interpolateOrder=0):
        assert (not self.training), 'Distance computation should happen in evaluation mode!'
        assert (input1.shape == input2.shape), 'Both inputs must have identical dimensions!'

        in1 = input1[None,...] if input1.ndim == 3 else input1
        in2 = input2[None,...] if input2.ndim == 3 else input2

        inputDict = {"reference": in1, "other": in2, "distance": 0.0, "path":""}
        data_transform = TransformsInference(interpolateTo, interpolateOrder, normMin=0, normMax=255)
        inputDict = data_transform(inputDict)

        inputDict["reference"] = torch.unsqueeze(inputDict["reference"], dim=1)
        inputDict["other"] = torch.unsqueeze(inputDict["other"], dim=1)

        output = self(inputDict)
        output = output.cpu().detach().view(-1).numpy()

        return output


    # ensures that feature map weights are greater or equal to zero
    def clampWeights(self):
        for linLayer in self.linear:
            for layer in linLayer:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    layer.weight.data = torch.clamp(layer.weight.data, min=0)


    # 1x1 convolution layer to scale feature maps channel-wise
    def linearLayer(self, channelsIn, featureMapSize):
        if self.useDropout:
            layer = nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(channelsIn, 1, 1, stride=1, padding=0, bias=False),
            )
            return layer
        else:
            layer = nn.Sequential(
                nn.Conv2d(channelsIn, 1, 1, stride=1, padding=0, bias=False),
            )
            return layer


    def initializeRandomSmall(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.01)
            if not m.bias is None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def initializeRandomLarge(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 1.0)
            if not m.bias is None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def initializeXavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            if not m.bias is None:
                m.bias.data.fill_(0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def initializeLayerwiseMean(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(m.weight.data.mean(), m.weight.data.std())
            if not m.bias is None:
                m.bias.data.fill_(0)


    # updates internal normalization accumulators for feature map normalization
    def updateNorm(self, sample):
        input1 = sample["reference"]
        input2 = sample["other"]

        if self.useGPU:
            input1 = input1.cuda()
            input2 = input2.cuda()

        sizeIn = input1.shape
        input1 = input1.view(sizeIn[0]*sizeIn[1], sizeIn[2], sizeIn[3], sizeIn[4])
        input2 = input2.view(sizeIn[0]*sizeIn[1], sizeIn[2], sizeIn[3], sizeIn[4])

        self.clampWeights()

        outBase1 = self.basenet(input1)
        outBase2 = self.basenet(input2)

        for i in range( len(outBase1) ):
            #print(outBase1[i].shape)
            normalized1 = self.normalizeTensor(outBase1[i], i, updateAcc=True)
            normalized2 = self.normalizeTensor(outBase2[i], i, updateAcc=True)


    # normalizes feature maps with different methods in channel dimension
    def normalizeTensor(self, tensorIn, layer, epsilon=1e-10, updateAcc=False):
        size = tensorIn.size()

        # unit normalize tensor
        if self.normMode == "normUnit":
            norm = torch.sqrt( torch.sum(tensorIn**2,dim=1) )
            norm = norm.view(size[0], 1, size[2], size[3])
            return tensorIn / (norm.expand_as(tensorIn) + epsilon)

        # normalize by max over all samples of all batches
        elif self.normMode == "normMax":
            if updateAcc:
                norm = torch.sqrt( torch.sum(tensorIn**2,dim=1) )
                temp = torch.max(norm, dim=0, keepdim=True)[0]
                self.normAcc[layer] = torch.max(self.normAcc[layer], temp)
            normMax = self.normAcc[layer]
            normMax = normMax.view(1, 1, size[2], size[3])
            return tensorIn / (normMax.expand_as(tensorIn) + epsilon)

        # normalize by avg over all samples of all batches
        elif self.normMode == "normAvg":
            if updateAcc:
                norm = torch.sqrt( torch.sum(tensorIn**2,dim=1) )
                temp = torch.sum(norm, dim=0)
                self.normAcc[layer] = (self.normAcc[layer] + temp)
                self.normCount[layer] = self.normCount[layer] + size[0]
            normAvg = self.normAcc[layer] / self.normCount[layer]
            normAvg = normAvg.view(1, 1, size[2], size[3])
            return tensorIn / (normAvg.expand_as(tensorIn) + epsilon)

        # create normal distribution for each feature map component by subtracting mean and dividing by the std. dev.
        # over all samples of all batches
        elif self.normMode == "normDist":
            if updateAcc:
                self.normCount[layer] = self.normCount[layer] + size[0]
                delta = tensorIn - self.normAcc[layer].detach().expand_as(tensorIn)
                self.normAcc[layer] = self.normAcc[layer].detach() + torch.sum( torch.mean(delta / self.normCount[layer], dim=1) , dim=0)
                self.normM2[layer] = self.normM2[layer].detach() + torch.sum( torch.mean(delta *(tensorIn - self.normAcc[layer].detach().expand_as(tensorIn)), dim=1) , dim=0)

            # rescale norm accumulators for differently sized inputs
            if size[2] != self.normAcc[layer].shape[0] or size[3] != self.normAcc[layer].shape[1]:
                up = nn.Upsample(size=(size[2], size[3]), mode="bilinear", align_corners=True)
                normAcc = torch.squeeze(up( torch.unsqueeze(torch.unsqueeze(self.normAcc[layer].detach(), dim=0), dim=0) ))
                normM2 = torch.squeeze(up( torch.unsqueeze(torch.unsqueeze(self.normM2[layer].detach(), dim=0), dim=0) ))

                mean = normAcc
                mean = mean.view(1, 1, size[2], size[3])
                std = torch.sqrt( normM2 / (self.normCount[layer] - 1) )
                std = std.view(1, 1, size[2], size[3])
            # directly use norm accumulators for input size 224x224
            else:
                mean = self.normAcc[layer].detach()
                mean = mean.view(1, 1, size[2], size[3])
                std = torch.sqrt( self.normM2[layer].detach() / (self.normCount[layer] - 1) )
                std = std.view(1, 1, size[2], size[3])
            normalized = (tensorIn - mean.expand_as(tensorIn)) / (std.expand_as(tensorIn) + epsilon)
            normalized = normalized / (math.sqrt(size[1]) - 1)
            return normalized

        # no normalization
        elif self.normMode == "normNone":
            return tensorIn
        else:
            raise ValueError('Unknown norm mode.')

    def printNumParams(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in parameters])
        print("Trainable parameters: %d" % params)

    # save model and normalization accumulators
    def save(self, path, override=False, noPrint=False):
        if not noPrint:
            print('Saving model to %s' % path)
        if not override and os.path.isfile(path):
            raise ValueError("Override warning!")
        else:
            if self.normMode != "normUnit":
                saveDict = {'stateDict' : self.state_dict(),
                            'normAcc' : self.normAcc,
                            'normM2' : self.normM2,
                            'normCount' : self.normCount, }
                torch.save(saveDict, path)
            else:
                torch.save(self.state_dict(), path)


    # load model and normalization accumulators
    def load(self, path):
        if self.normMode != "normUnit":
            if self.useGPU:
                print('Loading model from %s' % path)
                loaded = torch.load(path)
            else:
                print('CPU - Loading model from %s' % path)
                loaded = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(loaded['stateDict'])
            self.normAcc = loaded['normAcc']
            self.normM2 = loaded['normM2']
            self.normCount = loaded['normCount']
        else:
            if self.useGPU:
                print('Loading model from %s' % path)
                self.load_state_dict(torch.load(path))
            else:
                print('CPU - Loading model from %s' % path)
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


