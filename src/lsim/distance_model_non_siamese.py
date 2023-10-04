import os
import torch
import torch.nn as nn

from lsim.dataset_distance import *

class DistanceModelNonSiamese(nn.Module):

    def __init__(self, initBase="none", isTrain=False, useGPU=False):
        super(DistanceModelNonSiamese, self).__init__()
        self.useGPU = useGPU
        self.isTrain = isTrain

        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 12, stride=4, padding=2),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(32, 96, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(4, stride=2, padding=0),
            nn.Conv2d(96, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

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

        stacked = torch.cat([input1,input2], 1)
        feat = self.features(stacked)
        feat = feat.view(feat.size(0), 128 * 6 * 6)
        result = self.classifier(feat)
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


    def printNumParams(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in parameters])
        print("Trainable parameters: %d" % params)


    def save(self, path, override=False, noPrint=False):
        if not noPrint:
            print('Saving model to %s' % path)
        if not override and os.path.isfile(path):
            raise ValueError("Override warning!")
        else:
            torch.save(self.state_dict(), path)

    def load(self, path, useGPU=True):
        if useGPU:
            print('Loading model from %s' % path)
            self.load_state_dict(torch.load(path))
        else:
            print('CPU - Loading model from %s' % path)
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))