import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from lsim.distance_model import DistanceModel as LSIM_Model

from turbpred.params import LossParams


# input shape: B S C W H -> output shape: B S C
def loss_lsim(lsimModel:nn.Module, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    # normalize to [0,255] for each sequence
    xMin = torch.amin(x, dim=(1,3,4), keepdim=True)
    xMax = torch.amax(x, dim=(1,3,4), keepdim=True)
    yMin = torch.amin(y, dim=(1,3,4), keepdim=True)
    yMax = torch.amax(y, dim=(1,3,4), keepdim=True)
    ref = 255 * ((x - xMin) / (xMax - xMin))
    oth = 255 * ((y - yMin) / (yMax - yMin))
    ##dMin = torch.minimum(xMin, yMin)
    ##dMax = torch.maximum(xMax, yMax)
    ##ref = 255 * ((x - dMin) / (dMax - dMin))
    ##oth = 255 * ((y - dMin) / (dMax - dMin))

    # compare each sequence and channel individually by moving them to batch dimension
    # and adding a dummy channel dimension and lsim parameter dimension
    sizeBatch, sizeSeq, sizeChannel = ref.shape[0], ref.shape[1], ref.shape[2]
    ref = torch.reshape(ref, (-1,1,1,ref.shape[3], ref.shape[4]))
    oth = torch.reshape(oth, (-1,1,1,oth.shape[3], oth.shape[4]))

    # clone channel dimension as lsim compares 3-channel data
    ref = ref.expand(-1,-1,3,-1,-1)
    oth = oth.expand(-1,-1,3,-1,-1)

    inDict = {"reference": ref, "other": oth}
    distance = lsimModel(inDict)

    # move results from each original channel and sequence back into a
    # new channel and sequence dimension
    distance = torch.reshape(distance, (sizeBatch, sizeSeq, sizeChannel))

    return distance





class PredictionLoss(nn.modules.loss._Loss):

    def __init__(self, p_l:LossParams, dimension:int, simFields:List[str], useGPU:bool):
        super(PredictionLoss, self).__init__()
        self.p_l = p_l
        self.dimension = dimension
        self.simFields = simFields
        self.useGPU = useGPU

        if self.dimension == 2:
            # load lsim
            self.lsim = LSIM_Model(baseType="lsim", isTrain=False, useGPU=self.useGPU)
            self.lsim.load("src/lsim/models/LSiM.pth")
            self.lsim.eval()
            # freeze lsim weights
            for param in self.lsim.parameters():
                param.requires_grad = False


    def forward(self, prediction:torch.Tensor, groundTruth:torch.Tensor, latentSpace:torch.Tensor, vaeMeanVar:Tuple[torch.Tensor, torch.Tensor],
            weighted:bool=True, fadePredWeight:float=1, noLSIM:bool=False, ignorePredLSIMSteps:int=0) -> Tuple[torch.Tensor, dict, dict]:
        assert(not ((self.p_l.predMSE > 0) and prediction.shape[1] <= 1)), "Sequence length to small for prediction errors!"
        device = "cuda" if self.useGPU else "cpu"

        if not weighted:
            if self.dimension == 2:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4))
            elif self.dimension == 3:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4,5))
            lossRecMSE = seqMSE[0]
            lossPredMSE = torch.mean(seqMSE[1:])

            # only compute lsim loss on fields and ignore scalar simulation parameters
            numFields = self.dimension + len(self.simFields)
            if not noLSIM and self.dimension == 2:
                seqLSIM = loss_lsim(self.lsim, prediction[:,:,0:numFields], groundTruth[:,:,0:numFields]).mean((0,2))
            else:
                seqLSIM = torch.zeros(prediction.shape[0]).to(device)
                lossRecLSIM = torch.zeros(1).to(device)
                lossPredLSIM = torch.zeros(1).to(device)

            lossRecLSIM = seqLSIM[0]
            lossPredLSIM = torch.mean(seqLSIM[1:])

            lossRegMeanStd = torch.zeros(1).to(device) # not required in unweighted
            lossRegDiv = torch.zeros(1).to(device) # not required in unweighted
            lossRegVaeKLDiv = torch.zeros(1).to(device) # not required in unweighted
            lossRegLatStep = torch.zeros(1).to(device) # not required in unweighted

        else:
            lossRecMSE = torch.zeros(1).to(device)
            lossPredMSE = torch.zeros(1).to(device)
            lossRecLSIM = torch.zeros(1).to(device)
            lossPredLSIM = torch.zeros(1).to(device)
            lossRegMeanStd = torch.zeros(1).to(device)
            lossRegDiv = torch.zeros(1).to(device)
            lossRegVaeKLDiv = torch.zeros(1).to(device)
            lossRegLatStep = torch.zeros(1).to(device)

            if self.dimension == 2:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,2,3,4))
            elif self.dimension == 3:
                seqMSE = F.mse_loss(prediction, groundTruth, reduction="none").mean((0,3,4,5))
                if self.p_l.extraMSEvelZ > 0:
                    seqMSE[:,2:3] = self.p_l.extraMSEvelZ * seqMSE[:,2:3]
                seqMSE = torch.mean(seqMSE, dim=1)

            if self.p_l.recMSE > 0:
                lossRecMSE = self.p_l.recMSE * seqMSE[0]

            if self.p_l.predMSE > 0 and fadePredWeight > 0:
                lossPredMSE = self.p_l.predMSE * fadePredWeight * torch.mean(seqMSE[1:])

            # only compute lsim loss on fields and ignore scalar simulation parameters
            if self.dimension == 2:
                numFields = self.dimension + len(self.simFields)
                seqLSIM = None
                if self.p_l.predLSIM > 0:
                    if self.p_l.recLSIM > 0:
                        seqLSIM = loss_lsim(self.lsim, prediction[:,:,0:numFields], groundTruth[:,:,0:numFields]).mean((0,2))

                    else:
                        predLSIMStart = 1 + ignorePredLSIMSteps
                        seqLSIM = loss_lsim(self.lsim, prediction[:,predLSIMStart:,0:numFields], groundTruth[:,predLSIMStart:,0:numFields]).mean((0,2))
                        seqLSIM = torch.concat([torch.zeros(predLSIMStart, device=device), seqLSIM])

                elif self.p_l.recLSIM > 0:
                    seqLSIM = loss_lsim(self.lsim, prediction[:,0:1,0:numFields], groundTruth[:,0:1,0:numFields]).mean((0,2))

                if self.p_l.recLSIM > 0:
                    lossRecLSIM = self.p_l.recLSIM * seqLSIM[0]

                if self.p_l.predLSIM > 0 and fadePredWeight > 0:
                    lossPredLSIM = self.p_l.predLSIM * fadePredWeight * torch.mean(seqLSIM[1:])


            # mean and std regularization
            if self.p_l.regMeanStd > 0:
                if self.dimension == 2:
                    meanDiff = torch.abs(groundTruth.mean((3,4)) - prediction.mean((3,4)))
                    stdDiff  = torch.abs(groundTruth.std((3,4)) - prediction.std((3,4)))
                elif self.dimension == 3:
                    meanDiff = torch.abs(groundTruth.mean((3,4,5)) - prediction.mean((3,4,5)))
                    stdDiff  = torch.abs(groundTruth.std((3,4,5)) - prediction.std((3,4,5)))
                meanStd = meanDiff.mean() + stdDiff.mean()
                lossRegMeanStd = self.p_l.regMeanStd * meanStd

            # diveregence regularization
            if self.p_l.regMeanStd > 0:
                if self.dimension == 2:
                    vx_dx, _ = torch.gradient(prediction[:,:,0:1], dim=(3,4))
                    _, vy_dy = torch.gradient(prediction[:,:,1:2], dim=(3,4))
                    div = vx_dx + vy_dy
                elif self.dimension == 3:
                    vx_dx, _, _ = torch.gradient(prediction[:,:,0:1], dim=(3,4,5))
                    _, vy_dy, _ = torch.gradient(prediction[:,:,1:2], dim=(3,4,5))
                    _, _, vz_dz = torch.gradient(prediction[:,:,2:3], dim=(3,4,5))
                    div = vx_dx + vy_dy + vz_dz
                lossRegDiv = self.p_l.regDiv * torch.abs(div).mean()

            # KL divergence regularization for VAE
            if self.p_l.regVae > 0 and not vaeMeanVar[0] is None and not vaeMeanVar[1] is None:
                vaeMean = vaeMeanVar[0]
                vaeLogVar = vaeMeanVar[1]
                lossRegVaeKLDiv = -0.5 * torch.mean(1 + vaeLogVar - vaeMean.pow(2) - vaeLogVar.exp())
                lossRegVaeKLDiv = self.p_l.regVae * lossRegVaeKLDiv

            # latent space step regularization
            if self.p_l.regLatStep > 0 and latentSpace.shape[1] > 1:
                latFirst = latentSpace[:,0:latentSpace.shape[1]-1]
                latSecond = latentSpace[:,1:latentSpace.shape[1]]
                lossRegLatStep = self.p_l.regLatStep * torch.mean(torch.abs(latFirst - latSecond))

        loss = lossRecMSE + lossRecLSIM + lossPredMSE + lossPredLSIM + lossRegMeanStd + lossRegDiv + lossRegVaeKLDiv + lossRegLatStep
        lossParts = {
            "lossFull" : loss,
            "lossRecMSE" : lossRecMSE,
            "lossRecLSIM" : lossRecLSIM,
            "lossPredMSE" : lossPredMSE,
            "lossPredLSIM" : lossPredLSIM,
        }
        lossSeq = {"MSE" : seqMSE, "LSIM" : seqLSIM}
        return loss, lossParts, lossSeq
