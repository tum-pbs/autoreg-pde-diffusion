import torch
import torch.nn as nn
import torch.nn.functional as F

from turbpred.model_diffusion_blocks import Unet, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, cosine_beta_schedule
from turbpred.params import DataParams, ModelParamsDecoder
from turbpred.model_dfpnet import DfpNetTimeEmbedding


### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):
    p_d: DataParams
    p_md: ModelParamsDecoder

    def __init__(self, p_d:DataParams, p_md:ModelParamsDecoder, dimension:int, condChannels:int):
        super(DiffusionModel, self).__init__()

        self.p_d = p_d
        self.p_md = p_md
        self.dimension = dimension

        self.timesteps = self.p_md.diffSteps
        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = "random"        # "random" or "same"
        self.inferenceConditioningIntegration = self.p_md.diffCondIntegration # "noisy" or "clean"
        self.inferenceSamplingMode = "ddpm" if "ddpm" in self.p_md.arch else "ddim"

        if self.p_md.diffSchedule == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif self.p_md.diffSchedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        else:
            raise ValueError("Unknown variance schedule")
        
        betas = betas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (0,0,0,0,0,0,1,0), value=1.0)
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1. - alphasCumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posteriorVariance = betas * (1. - alphasCumprodPrev) / (1. - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrtRecipAlphas", sqrtRecipAlphas)
        self.register_buffer("sqrtAlphasCumprod", sqrtAlphasCumprod)
        self.register_buffer("sqrtOneMinusAlphasCumprod", sqrtOneMinusAlphasCumprod)
        self.register_buffer("sqrtPosteriorVariance", sqrtPosteriorVariance)

        if "dfp" in self.p_md.arch:
            self.unet = DfpNetTimeEmbedding(
                inChannels= condChannels + (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)),
                outChannels= condChannels + (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)),
                blockChannels=self.p_md.decWidth,
            )
        else:
            self.unet = Unet(
                dim=self.p_d.dataSize[0],
                channels= condChannels + (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)),
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1,
            )


    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning:torch.Tensor, data:torch.Tensor) -> torch.Tensor:
        if self.dimension == 3:
            raise NotImplementedError()

        device = "cuda" if data.is_cuda else "cpu"
        seqLen = data.shape[1]

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))

        # TRAINING
        if self.training:

            # forward diffusion process that adds noise to data
            if self.p_md.diffCondIntegration == "noisy":
                d = torch.concat((cond, d), dim=1)
                noise = torch.randn_like(d, device=device)
                t = torch.randint(0, self.timesteps, (d.shape[0],), device=device).long()
                dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise

            elif self.p_md.diffCondIntegration == "clean":
                dNoise = torch.randn_like(d, device=device)
                t = torch.randint(0, self.timesteps, (d.shape[0],), device=device).long()
                dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise

                noise = torch.concat((cond, dNoise), dim=1)
                dNoisy = torch.concat((cond, dNoisy), dim=1)

            else:
                raise ValueError("Unknown conditioning integration mode")


            # noise prediction with network
            predictedNoise = self.unet(dNoisy, t)

            # unstack batch and sequence dimension again
            noise = torch.reshape(noise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))
            predictedNoise = torch.reshape(predictedNoise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))

            return noise, predictedNoise


        # INFERENCE
        else:
            # conditioned reverse diffusion process
            if self.inferenceInitialSampling == "random":
                dNoise = torch.randn_like(d, device=device)
                cNoise = torch.randn_like(cond, device=device)
            else:
                dNoise = torch.randn((1, d.shape[1], d.shape[2], d.shape[3]), device=device).expand(d.shape[0],-1,-1,-1)
                cNoise = torch.randn((1, cond.shape[1], cond.shape[2], cond.shape[3]), device=device).expand(cond.shape[0],-1,-1,-1)

            sampleStride = 1
            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                if self.inferenceConditioningIntegration == "noisy":
                    condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise
                else:
                    condNoisy = cond

                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                # backward diffusion process that removes noise to create data
                predictedNoiseCond = self.unet(dNoiseCond, t)

                # use model (noise predictor) to predict mean
                modelMean = self.sqrtRecipAlphas[t] * (dNoiseCond - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]] # discard prediction of conditioning
                if i != 0 and self.inferenceSamplingMode == "ddpm":
                    if self.inferencePosteriorSampling == "random":
                        # sample randomly (only for non-final prediction)
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)
                    else:
                        # sample with same seed (only for non-final prediction)
                        postNoise = torch.randn((1, dNoise.shape[1], dNoise.shape[2], dNoise.shape[3]), device=device).expand(dNoise.shape[0],-1,-1,-1)
                        dNoise = dNoise + self.sqrtPosteriorVariance[t] * postNoise

            # unstack batch and sequence dimension again
            dNoise = torch.reshape(dNoise, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return dNoise