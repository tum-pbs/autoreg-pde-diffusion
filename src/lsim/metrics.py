import torch
import torch.nn as nn
import numpy as np
import skimage.metrics as metrics

from lsim.dataset_distance import *

class Metric(nn.Module):
    def __init__(self, mode):
        super(Metric, self).__init__()
        assert (mode in ["L2", "SSIM", "PSNR", "MI"]), "Unknown metric mode!"
        self.mode = mode
        self.eval()

    def forward(self, x):
        input1 = x["reference"].cpu().numpy().astype(np.uint8)
        input2 = x["other"].cpu().numpy().astype(np.uint8)

        sizeIn = input1.shape

        distance = np.empty( (sizeIn[0], sizeIn[1]) )
        for i in range(sizeIn[0]):
            for j in range(sizeIn[1]):
                in1 = np.transpose(input1[i,j], [1,2,0])
                in2 = np.transpose(input2[i,j], [1,2,0])
                if self.mode == "L2":
                    distance[i,j] = metrics.mean_squared_error(in1, in2) / (255.0 * 255.0)
                elif self.mode == "SSIM":
                    distance[i,j] =  1 - metrics.structural_similarity(in1, in2, multichannel=True) #invert as distance measure
                elif self.mode == "PSNR":
                    distance[i,j] =  -metrics.peak_signal_noise_ratio(in1, in2) #invert as distance measure
                elif self.mode == "MI":
                    distance[i,j] =  np.mean(metrics.variation_of_information(in1, in2))
        return torch.from_numpy(distance)


    # input two 4D/3D numpy arrays in order [batch, width, height, channels] or
    # [width, height, channels] and return a distance of shape [batch] or [1]
    def computeDistance(self, input1, input2):
        assert (input1.shape == input2.shape), 'Both inputs must have identical dimensions!'

        in1 = input1[None,...] if input1.ndim == 3 else input1
        in2 = input2[None,...] if input2.ndim == 3 else input2

        inputDict = {"reference": in1, "other": in2, "distance": 0.0, "path":""}
        data_transform = TransformsInference(None, None, normMin=0, normMax=255)
        inputDict = data_transform(inputDict)

        inputDict["reference"] = torch.unsqueeze(inputDict["reference"], dim=1)
        inputDict["other"] = torch.unsqueeze(inputDict["other"], dim=1)

        output = self(inputDict)
        output = output.cpu().detach().view(-1).numpy()

        return output