from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import torch
import os
import re
import random
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import math

class DatasetDistance(Dataset):
    def __init__(self, name, dataDirs, exclude=[], include=[], transform=None, fileType="png"):
        self.transform = transform
        self.name = name
        self.fileType = fileType
        self.dataPaths = []

        print("Dataset " + name + " at " + str(dataDirs))

        for dataDir in dataDirs:
            directories = os.listdir(dataDir)
            directories.sort()
            for directory in directories:
                if exclude:
                    if any( item in directory for item in exclude ) :
                        continue
                if include:
                    if not any( item in directory for item in include ) :
                        continue

                currentDir = os.path.join(dataDir, directory)
                self.dataPaths.append(currentDir)

        print("Length: %d" % len(self.dataPaths))

    def __len__(self):
        return len(self.dataPaths)

    def __getitem__(self, idx):
        directory = self.dataPaths[idx]
        fileNames = os.listdir(directory)
        fileNames.sort()

        listFrames = []
        listDist = []

        for fileName in fileNames:
            filePath = os.path.join(directory, fileName)
            if not fileName.endswith(".%s" % self.fileType) or fileName == "ref.%s" % self.fileType:
                continue

            if self.fileType == "png":
                frame = imageio.imread(filePath)
            elif self.fileType == "npz":
                frame = np.load(filePath)['arr_0']

            if frame.ndim == 2:
                frame = frame[...,None]
            if frame.shape[2] == 4:
                frame = frame[...,:-1]

            start = len(fileName) - 6
            end = len(fileName) - 4
            listDist.append( float(fileName[start:end]) )
            listFrames.append(frame)

        assert(len(listDist) != 0 and len(listFrames) != 0), "%s: no data to load!" % directory

        distances = np.array(listDist)
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

        frames = np.array(listFrames)
        #if frames.shape[0] != 11:
        #    print("%s is missing files!" % directory)

        reference = []
        other = []
        dist = []

        for i in range(distances.shape[0]):
            for j in range(i+1,distances.shape[0]):
                diff = distances[j] - distances[i]
                if diff < 0:
                    raise ValueError('Training distances have to be positive!')
                reference.append(frames[i])
                other.append(frames[j])
                dist.append(diff)

        sample = {"reference": np.stack(reference, 0), "other": np.stack(other, 0),
                "distance": np.stack(dist, 0), "path": directory}
            
        if self.transform:
            sample = self.transform(sample)
        return sample


    def setDataTransform(self, transform):
        self.transform = transform

    def computeMeanAndStd(self):
        print("Computing mean and std of dataset...")
        mean = 0 # online data mean
        count = 0 # accumulator for computing online standard deviation
        M2 = 0 # accumulator for computing online standard deviation

        for path in self.dataPaths:
            fileNames = os.listdir(path)
            fileNames.sort()

            for fileName in fileNames:
                filePath = os.path.join(path, fileName)
                if not fileName.endswith(".%s" % self.fileType) or fileName == "ref.%s" % self.fileType:
                    continue

                if self.fileType == "png":
                    data = imageio.imread(filePath)
                elif self.fileType == "npz":
                    data = np.load(filePath)['arr_0']

                if data.shape[2] == 4:
                    data = data[...,:-1]

                count += data.shape[0] * data.shape[1] * data.shape[2]
                delta = data - mean
                mean += np.sum(delta / count)
                M2 += np.sum(delta * (data - mean))

        std = np.sqrt(M2 / (count-1))

        self.mean, self.std = [mean, std]

        return mean, std

# ------------------------------------------------- 
# TRANSFORMS TO APPLY TO THE DATA
# -------------------------------------------------

# combines randomFlip, randomRotation90, randomCrop,
# channelSwap, toTensor and normalization for efficiency
class TransformsTrain(object):
    def __init__(self, outputSize, normMin=0, normMax=255):
        self.outputSize = outputSize
        self.normMin = normMin
        self.normMax = normMax
        self.angles = [0,1,2,3]

    def __call__(self, sample):
        dist = sample["distance"]
        reference = sample["reference"]
        other = sample["other"]
        path = sample["path"]

        resultRef = np.zeros([reference.shape[0], self.outputSize, self.outputSize, reference.shape[3]])
        resultOther = np.zeros([other.shape[0], self.outputSize, self.outputSize, other.shape[3]])

        for i in range(reference.shape[0]):
            ref = reference[i]
            oth = other[i]

            # flip
            rand = random.randint(0, 3)
            if rand == 1:
                ref = np.fliplr(ref)
                oth = np.fliplr(oth)
            if rand == 2:
                ref = np.flipud(ref)
                oth = np.flipud(oth)
            if rand == 3:
                ref = np.flipud( np.fliplr(ref) )
                oth = np.flipud( np.fliplr(oth) )

            # rot90
            angle = random.choice(self.angles)
            ref = np.rot90(ref, angle)
            oth = np.rot90(oth, angle)

            # channel swap
            channelOrder = [0,1,2]
            random.shuffle(channelOrder)
            ref = ref[..., channelOrder]
            oth = oth[..., channelOrder]

            # crop
            if ref.shape[0] <= self.outputSize or ref.shape[1] <= self.outputSize:
                resultRef[i] = ref
                resultOther[i] = oth
                continue

            top = np.random.randint(0, ref.shape[0] - self.outputSize)
            left = np.random.randint(0, ref.shape[1] - self.outputSize)

            resultRef[i] = ref[top : top+self.outputSize,  left : left+self.outputSize]
            resultOther[i] = oth[top : top+self.outputSize,  left : left+self.outputSize]

        # normalization
        dMin = np.minimum( np.min(resultRef, axis=(0,1,2)), np.min(resultOther, axis=(0,1,2)) )
        dMax = np.maximum( np.max(resultRef, axis=(0,1,2)), np.max(resultOther, axis=(0,1,2)) )
        if (dMin == dMax).all():
            resultRef = resultRef - dMin
            resultOther = resultOther - dMin
        elif (dMin == dMax).any():
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    resultRef[..., i] = resultRef[..., i] - dMin[i]
                    resultOther[..., i] = resultOther[..., i] - dMin[i]
                else:
                    resultRef[..., i] = (self.normMax - self.normMin) * ( (resultRef[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
                    resultOther[..., i] = (self.normMax - self.normMin) * ( (resultOther[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
        else:
            resultRef = (self.normMax - self.normMin) * ( (resultRef - dMin) / (dMax - dMin) ) + self.normMin
            resultOther = (self.normMax - self.normMin) * ( (resultOther - dMin) / (dMax - dMin) ) + self.normMin

        # toTensor
        resultRef = torch.from_numpy(resultRef.transpose(0,3,1,2)).float()
        resultOther = torch.from_numpy(resultOther.transpose(0,3,1,2)).float()
        dist = torch.from_numpy(np.array(dist)).float()

        return {"reference": resultRef, "other": resultOther, "distance": dist, "path": path}


# combines resize, toTensor and normalization for efficiency
class TransformsInference(object):
    def __init__(self, outputSize, order, normMin = 0, normMax = 255):
        self.normMin = normMin
        self.normMax = normMax
        self.outputSize = outputSize
        self.order = order

    def __call__(self, sample):
        dist = sample["distance"]
        reference = sample["reference"]
        other = sample["other"]
        path = sample["path"]

        # repeat for scalar fields
        if reference.shape[reference.ndim-1] == 1:
            reference = np.repeat(reference, 3, axis=reference.ndim-1)
        if other.shape[other.ndim-1] == 1:
            other = np.repeat(other, 3, axis=other.ndim-1)

        # resize
        if self.outputSize and (self.outputSize != reference.shape[1] or self.outputSize != reference.shape[2]):
            resultRef = np.zeros([reference.shape[0], self.outputSize, self.outputSize, reference.shape[3]])
            resultOther = np.zeros([other.shape[0], self.outputSize, self.outputSize, other.shape[3]])

            zoom1 = [1, self.outputSize / reference.shape[1], self.outputSize / reference.shape[2], 1]
            resultRef = scipy.ndimage.zoom(reference, zoom1, order=self.order)
            zoom2 = [1, self.outputSize / other.shape[1], self.outputSize / other.shape[2], 1]
            resultOther = scipy.ndimage.zoom(other, zoom2, order=self.order) 
        else:
            resultRef = reference
            resultOther = other

        # normalization
        dMin = np.minimum( np.min(resultRef, axis=(0,1,2)), np.min(resultOther, axis=(0,1,2)) )
        dMax = np.maximum( np.max(resultRef, axis=(0,1,2)), np.max(resultOther, axis=(0,1,2)) )
        if (dMin == dMax).all():
            resultRef = resultRef - dMin
            resultOther = resultOther - dMin
        elif (dMin == dMax).any():
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    resultRef[..., i] = resultRef[..., i] - dMin[i]
                    resultOther[..., i] = resultOther[..., i] - dMin[i]
                else:
                    resultRef[..., i] = (self.normMax - self.normMin) * ( (resultRef[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
                    resultOther[..., i] = (self.normMax - self.normMin) * ( (resultOther[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
        else:
            resultRef = (self.normMax - self.normMin) * ( (resultRef - dMin) / (dMax - dMin) ) + self.normMin
            resultOther = (self.normMax - self.normMin) * ( (resultOther - dMin) / (dMax - dMin) ) + self.normMin

        # toTensor
        resultRef = torch.from_numpy(resultRef.transpose(0,3,1,2)).float()
        resultOther = torch.from_numpy(resultOther.transpose(0,3,1,2)).float()
        dist = torch.from_numpy(np.array(dist)).float()

        return {"reference": resultRef, "other": resultOther, "distance": dist, "path": path}
