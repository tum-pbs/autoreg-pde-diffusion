import torch
import torch.nn as nn
import math

class CorrelationLoss(nn.modules.loss._Loss):
    def __init__(self, weightMSE, weightCorr, weightCrossCorr):
        super(CorrelationLoss, self).__init__()
        self.weightMSE = weightMSE
        self.weightCorr = weightCorr
        self.weightCrossCorr = weightCrossCorr

        self.lastForwardCor = -1
        self.lastForwardLossCor = 0
        self.lastForwardLossL2 = 0

    def forward(self, outputDist, targetDist):
        # pearson correlation
        cor = self.corrcoef(outputDist[:,:10], targetDist[:,:10])
        correlation = self.weightCorr * 0.5 * (1-cor)

        # mse
        l2 = self.weightMSE * self.distanceL2(outputDist, targetDist)

        # cross correlation
        crossed = self.crossCorr(outputDist[:,:10], targetDist[:,:10])
        cross = self.weightCrossCorr * 0.5 * (1-crossed)

        self.lastForwardCor = torch.sum(cor).item()
        self.lastForwardLossCor = torch.sum(correlation).item()
        self.lastForwardLossL2 = torch.sum(l2).item()
        return torch.sum(l2 + correlation + cross)


    # Implemented similar to scipy.stats.pearsonr, see https://github.com/pytorch/pytorch/issues/1254
    # input: 2 1D torch tensors of same shape   output: correlation coefficient
    def corrcoef(self, x, y):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = torch.sum(xm * ym, dim=1, keepdim=True)  # manual dot product
        r_den = torch.norm(xm, 2, dim=1, keepdim=True) * torch.norm(ym, 2, dim=1, keepdim=True)
        r_val = r_num / r_den
        return r_val

    def crossCorr(self, x, y):
        dot = torch.sum(x * y, dim=1, keepdim=True)  # manual dot product
        mag = torch.norm(x, 2, dim=1, keepdim=True) * torch.norm(y, 2, dim=1, keepdim=True)
        return dot / mag

    def distanceL2(self, x, y):
        result = torch.pow(y - x, 2)
        return torch.mean(result, dim=1, keepdim=True)

    def distanceL1(self, x, y):
        result = torch.abs(y - x)
        return torch.mean(result, dim=1, keepdim=True)