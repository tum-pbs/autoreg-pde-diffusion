import torch
import torch.nn as nn
import torch.nn.functional as F

from turbpred.model_diffusion_blocks import Unet
from turbpred.params import DataParams, ModelParamsDecoder


class PDERefiner(nn.Module):
    def __init__(self, p_d:DataParams, p_md:ModelParamsDecoder, condChannels:int):
        super(PDERefiner, self).__init__()

        self.p_d = p_d
        self.p_md = p_md

        self.numSteps = self.p_md.diffSteps
        self.minNoiseStd = self.p_md.refinerStd

        self.unet = Unet(
            dim=self.p_d.dataSize[0],
            channels= condChannels + (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)),
            out_dim= self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams),
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
        )


    def forward(self, conditioning:torch.Tensor, data:torch.Tensor) -> torch.Tensor:
        device = "cuda" if data.is_cuda else "cpu"
        seqLen = data.shape[1]

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))

        # TRAINING
        if self.training:
            k = torch.randint(0, self.numSteps, (d.shape[0],1,1,1), device=device).long()
            kSqueeze = k.squeeze(3).squeeze(2).squeeze(1)

            # add noise to data according to sampled k
            noiseStd = self.minNoiseStd ** (k / self.numSteps)
            noise = torch.randn_like(d, device=device)
            dataNoisy = d + noise * noiseStd

            # mask model input to zero where k == 0
            dataNoisyMasked = torch.where(k == 0, torch.zeros_like(dataNoisy), dataNoisy)
            pred = self.unet( torch.concat((cond, dataNoisyMasked), dim=1), kSqueeze)

            # adjust prediction target according to k
            target = torch.where(k == 0, d, noise)

            # unstack batch and sequence dimension again
            pred = torch.reshape(pred, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))
            target = torch.reshape(target, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return pred, target

        # INFERENCE
        else:
            # first step is direct prediction
            kZero = torch.zeros(cond.shape[0], device=device).long()
            pred = self.unet( torch.concat((cond, torch.zeros_like(d)), dim=1), kZero)
            for i in range(1, self.numSteps + 1):
                k = i * torch.ones((cond.shape[0],1,1,1), device=device).long()
                kSqueeze = k.squeeze(3).squeeze(2).squeeze(1)

                # add noise to data according to k
                noiseStd = self.minNoiseStd ** (k / self.numSteps)
                noise = torch.randn_like(d, device=device)
                predNoisy = pred + noise * noiseStd

                predictedNoise = self.unet( torch.concat((cond, predNoisy), dim=1), kSqueeze)
                pred = predNoisy - predictedNoise * noiseStd

            # unstack batch and sequence dimension again
            pred = torch.reshape(pred, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))
            return pred

