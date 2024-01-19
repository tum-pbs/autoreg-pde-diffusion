import numpy as np
import logging, os
import torch
import torch.nn as nn

from neuralop.models import FNO

from turbpred.model_encoder import EncoderModelSkip, DecoderModelSkip
from turbpred.model_latent_transformer import LatentModelTransformerEnc, LatentModelTransformerDec, LatentModelTransformer, LatentModelTransformerMGN, LatentModelTransformerMGNParamEmb
from turbpred.model_diffusion import DiffusionModel
from turbpred.model_diffusion_blocks import Unet
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent

from turbpred.model_dfpnet import DfpNet
from turbpred.model_resnet import DilatedResNet
from turbpred.model_refiner import PDERefiner

class PredictionModel(nn.Module):
    p_d: DataParams
    p_t: TrainingParams
    p_l: LossParams
    p_me: ModelParamsEncoder
    p_md: ModelParamsDecoder
    p_ml: ModelParamsLatent
    useGPU: bool

    def __init__(self, p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
                p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True):
        super(PredictionModel, self).__init__()

        self.p_d = p_d
        self.p_t = p_t
        self.p_l = p_l
        self.p_me = p_me
        self.p_md = p_md
        self.p_ml = p_ml
        self.useGPU = useGPU

        if (self.p_me and self.p_me.pretrained) or (self.p_md and self.p_md.pretrained) or (self.p_ml and self.p_ml.pretrained):
            if pretrainPath:
                loadedPretrainedWeightDict = torch.load(pretrainPath)

        # ENCODER
        if self.p_me:
            if self.p_me.arch == "skip":
                self.modelEncoder = EncoderModelSkip(p_d, p_me, p_ml, p_d.dimension)
            else:
                raise ValueError("Unknown encoder architecture!")

            # load pretraining weights
            if pretrainPath and self.p_me.pretrained:
                self.modelEncoder.load_state_dict(loadedPretrainedWeightDict["stateDictEncoder"])

            # freeze weights
            if self.p_me.frozen:
                for param in self.modelEncoder.parameters():
                    param.requires_grad = False
        else:
            self.modelEncoder = None


        # DECODER
        if self.p_md:
            if self.p_md.arch == "skip":
                self.modelDecoder = DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension)

            elif self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                                    "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                                    "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                                    "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                                    "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",]:
                if "+Prev" in self.p_md.arch:
                    prevSteps = 2
                elif "+2Prev" in self.p_md.arch:
                    prevSteps = 3
                elif "+3Prev" in self.p_md.arch:
                    prevSteps = 4
                else:
                    prevSteps = 1

                inChannels = prevSteps * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                outChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)

                if "unet" in self.p_md.arch:
                    self.modelDecoder = Unet(dim=self.p_d.dataSize[0], out_dim=outChannels, channels=inChannels,
                                        dim_mults=(1,1,1), use_convnext=True, convnext_mult=1, with_time_emb=False)

                elif "resnet" in self.p_md.arch:
                    self.modelDecoder = DilatedResNet(inFeatures=inChannels, outFeatures=outChannels, blocks=4, features=self.p_md.decWidth, dilate="dil_" in self.p_md.arch)

                elif "fno" in self.p_md.arch:
                    self.modelDecoder = FNO(n_modes=(self.p_md.fnoModes[0],self.p_md.fnoModes[1]), hidden_channels=self.p_md.decWidth, in_channels=inChannels, out_channels=outChannels, n_layers=4)

                elif "dfp" in self.p_md.arch:
                    self.modelDecoder = DfpNet(inChannels=inChannels, outChannels=outChannels, blockChannels=self.p_md.decWidth)

                else:
                    raise ValueError("Unknown decoder architecture")


            elif self.p_md.arch in ["decode-ddpm", "decode-ddim", "direct-ddpm+First", "direct-ddim+First",
                                    "direct-ddpm", "direct-ddim", "direct-ddpm+Prev", "direct-ddim+Prev",
                                    "direct-ddpm+2Prev", "direct-ddim+2Prev", "direct-ddpm+3Prev", "direct-ddim+3Prev", 
                                    "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",
                                    "direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                if self.p_md.arch in ["decode-ddpm", "decode-ddim"]:
                    condChannels = self.p_me.latentSize + len(self.p_d.simParams)
                elif self.p_md.arch in ["direct-ddpm", "direct-ddim", "dfp-ddpm"]:
                    condChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)
                elif self.p_md.arch in ["direct-ddpm+First", "direct-ddim+First", "direct-ddpm+Prev", "direct-ddim+Prev", "dfp-ddpm+Prev"]:
                    condChannels = 2 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+2Prev", "direct-ddim+2Prev", "dfp-ddpm+2Prev"]:
                    condChannels = 3 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+3Prev", "direct-ddim+3Prev", "dfp-ddpm+3Prev"]:
                    condChannels = 4 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                elif self.p_md.arch in ["direct-ddpm+Enc", "direct-ddim+Enc", "hybrid-ddpm+Lat", "hybrid-ddim+Lat"]:
                    condChannels = (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)) + (self.p_me.latentSize + len(self.p_d.simParams))
                self.modelDecoder = DiffusionModel(p_d, p_md, p_d.dimension, condChannels=condChannels)

            elif self.p_md.arch == "refiner":
                condChannels = self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)
                self.modelDecoder = PDERefiner(p_d, p_md, condChannels=condChannels)


            elif self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams))
                ])

            elif self.p_md.arch in ["skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                self.modelDecoder = nn.ModuleList([
                    DecoderModelSkip(p_d, p_me, p_md, p_ml, p_d.dimension),
                    DiffusionModel(p_d, p_md, p_d.dimension, condChannels=2 * (self.p_d.dimension + len(self.p_d.simFields) + len(self.p_d.simParams)))
                ])

            else:
                raise ValueError("Unknown decoder architecture!")

            # load pretraining weights
            if pretrainPath and self.p_md.pretrained:
                if self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    self.modelDecoder[0].load_state_dict(loadedPretrainedWeightDict["stateDictDecoder"])
                else:
                    self.modelDecoder.load_state_dict(loadedPretrainedWeightDict["stateDictDecoder"])

            # freeze weights
            if self.p_md.frozen:
                if self.p_md.arch in ["skip+finetune-ddpm", "skip+finetune-ddim", "skip+hybrid-ddpm", "skip+hybrid-ddim"]:
                    for param in self.modelDecoder[0].parameters():
                        param.requires_grad = False
                else:
                    for param in self.modelDecoder.parameters():
                        param.requires_grad = False
        else:
            self.modelDecoder = None

        # LATENT MODEL
        if self.p_ml:
            if self.p_ml.arch == "transformerEnc":
                self.modelLatent = LatentModelTransformerEnc(p_d, p_me, p_ml, False)
            elif self.p_ml.arch == "transformerDec":
                self.modelLatent = LatentModelTransformerDec(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformer":
                self.modelLatent = LatentModelTransformer(p_d, p_me, p_ml)
            elif self.p_ml.arch == "transformerMGN":
                self.modelLatent = LatentModelTransformerMGN(p_d, p_me, p_ml)
                self.modelLatentParamEmb = LatentModelTransformerMGNParamEmb(p_d, p_me)
            else:
                raise ValueError("Unknown latent architecture!")

            # load pretraining weights
            if pretrainPath and self.p_ml.pretrained:
                self.modelLatent.load_state_dict(loadedPretrainedWeightDict["stateDictLatent"])

            # freeze weights
            if self.p_ml.frozen:
                for param in self.modelLatent.parameters():
                    param.requires_grad = False
        else:
            self.modelLatent = None

        self.to("cuda" if self.useGPU else "cpu")



    def forward(self, data:torch.Tensor, simParameters:torch.Tensor, useLatent:bool=True) -> torch.Tensor:
        device = "cuda" if self.useGPU else "cpu"
        d = data.to(device)
        simParam = simParameters.to(device) if simParameters is not None else None

        # ENCODING - LATENT MODEL - DECODING
        if not (self.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",
                "refiner",
                "direct-ddpm", "direct-ddim", "direct-ddpm+First", "direct-ddim+First", 
                "direct-ddpm+Prev", "direct-ddim+Prev", "direct-ddpm+2Prev", "direct-ddim+2Prev",
                "direct-ddpm+3Prev", "direct-ddim+3Prev", "direct-ddpm+Enc", "direct-ddim+Enc",
                "dfp-ddpm", "dfp-ddpm+Prev", "dfp-ddpm+2Prev", "dfp-ddpm+3Prev",]):

            latentSpace = torch.zeros(d.shape[0], d.shape[1], self.p_me.latentSize)
            latentSpace = latentSpace.to(device)
            if not self.modelLatent or not useLatent:
                # no latent model -> fully process sequence with AE
                latentSpace = self.modelEncoder(d)
            else:
                if isinstance(self.modelLatent, LatentModelTransformerEnc):
                    latentSpace = self.forwardTransEnc(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerDec) or isinstance(self.modelLatent, LatentModelTransformer):
                    latentSpace = self.forwardTransDec(d, latentSpace, simParam)

                elif isinstance(self.modelLatent, LatentModelTransformerMGN):
                    latentSpace = self.forwardTransMGN(d, latentSpace, simParam)

                else:
                    raise ValueError("Invalid latent model!")

            if "decode" in self.p_md.arch:
                prediction = self.forwardDiffusionDecode(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "finetune" in self.p_md.arch:
                prediction = self.forwardDiffusionFinetune(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            elif "hybrid" in self.p_md.arch:
                prediction = self.forwardDiffusionHybrid(d, latentSpace, simParam)
                return prediction, latentSpace, (None, None)

            else:
                prediction, vaeMeanVar = self.modelDecoder(latentSpace, simParam)
                return prediction, latentSpace, vaeMeanVar


        # DIRECT PREDICTION OF NEXT FRAME WITH DIFFERENT ARCHITECTURES
        else:
            if isinstance(self.modelDecoder, Unet) or isinstance(self.modelDecoder, DilatedResNet) or isinstance(self.modelDecoder, FNO):
                prediction = self.forwardDirect(d)
                return prediction, None, (None, None)

            else:
                prediction = self.forwardDiffusionDirect(d, simParam)
                return prediction, None, (None, None)




    # Transformer encoder latent model
    def forwardTransEnc(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer encoder latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace, simParam)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace[:,:-1]], dim=1)

        # transformer encoder latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                transLatentSpace = self.modelLatent(latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None)
                latentSpace[:,i] = transLatentSpace[:,-1]

        return latentSpace


    # Transformer decoder latent model
    def forwardTransDec(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeSeq = d.shape[1]

        # transformer latent model to predict single next step
        if self.training and not self.p_ml.transTrainUnroll:
            encLatentSpace = self.modelEncoder(d)
            transLatentSpace = self.modelLatent(encLatentSpace[:,:-1], encLatentSpace[:,1:], simParam[:,:-1] if simParam is not None else None, simParam[:,1:] if simParam is not None else None)
            latentSpace = torch.concat([encLatentSpace[:,:1], transLatentSpace], dim=1)

        # transformer latent model to predict all steps from first one
        else:
            latentSpace[:,0:1] = self.modelEncoder(d[:,0:1])
            for i in range(1,sizeSeq):
                if self.p_ml.transTargetFull:
                    start = max(0, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 0
                    transLatentSpace = self.modelLatent(latentSpace[:,start:i], latentSpace[:,start:i], simParam[:,start:i] if simParam is not None else None, simParam[:,start:i] if simParam is not None else None)
                    latentSpace[:,i] = transLatentSpace[:,-1]
                    #latentSpace[:,i] = latentSpace[:,i-1] + transLatentSpace[:,-1]
                else:
                    transLatentSpace = self.modelLatent(latentSpace[:,:i], latentSpace[:,i-1:i], simParam[:,:i] if simParam is not None else None, simParam[:,i-1:i] if simParam is not None else None)
                    latentSpace[:,i:i+1] = transLatentSpace

        return latentSpace


    # Transformer latent model according to MeshGraphNet paper
    def forwardTransMGN(self, d:torch.Tensor, latentSpace:torch.Tensor, simParam:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        latentSpace = torch.zeros(sizeBatch, sizeSeq+1, self.p_me.latentSize)
        latentSpace = latentSpace.to("cuda" if self.useGPU else "cpu")

        if simParam is not None:
            latentSpace[:,0] = self.modelLatentParamEmb(simParam[:,0]) # only use scalar simParam input
        latentSpace[:,1:2] = self.modelEncoder(d[:,0:1])
        for i in range(2,sizeSeq+1):
            start = max(1, i-self.p_ml.maxInputLen) if self.p_ml.maxInputLen > 0 else 1
            transInput = torch.concat([latentSpace[:,0:1], latentSpace[:,start:i]], dim=1)
            transLatentSpace = self.modelLatent(transInput, latentSpace[:,i-1:i])
            latentSpace[:,i:i+1] = latentSpace[:,i-1:i] + transLatentSpace
        latentSpace = latentSpace[:,1:] # discard param embedding

        return latentSpace


    # Direct prediction of next step via U-Net, ResNet, FNO, etc.
    def forwardDirect(self, d:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        if "+Prev" in self.p_md.arch:
            prevSteps = 2
        elif "+2Prev" in self.p_md.arch:
            prevSteps = 3
        elif "+3Prev" in self.p_md.arch:
            prevSteps = 4
        else:
            prevSteps = 1

        prediction = []
        #for i in range(4):
        for i in range(prevSteps): # no prediction of first steps
            if self.training:
                trainNoise = self.p_md.trainingNoise * torch.normal(torch.zeros_like(d[:,i]), torch.ones_like(d[:,i]))
                prediction += [d[:,i] + trainNoise]
            else:
                prediction += [d[:,i]]

        for i in range(prevSteps, sizeSeq):
            uIn = torch.concat(prediction[i-prevSteps : i], dim=1)

            if isinstance(self.modelDecoder, FNO):
                result = self.modelDecoder(uIn)
            else:
                result = self.modelDecoder(uIn, None)
    
            if self.p_d.simParams:
                result[:,-len(self.p_d.simParams):] = d[:,i,-len(self.p_d.simParams):] # replace simparam prediction with true values
            prediction += [result]

        prediction = torch.stack(prediction, dim=1)
        return prediction



    # Diffusion model to directly predict next step based on different conditionings
    def forwardDiffusionDirect(self, d:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        # TRAINING
        if self.training:
            if "+Enc" in self.p_md.arch:
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((conditioning, d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex]

            elif "+First" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq, (1,), device=d.device)
                conditioning = torch.concat((d[:,0:1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

            else:
                if "+Prev" in self.p_md.arch:
                    prevSteps = 2
                elif "+2Prev" in self.p_md.arch:
                    prevSteps = 3
                elif "+3Prev" in self.p_md.arch:
                    prevSteps = 4
                else:
                    prevSteps = 1

                cond = []
                for i in range(prevSteps):
                    trainNoise = self.p_md.trainingNoise * torch.normal(torch.zeros_like(d[:,i:i+1]), torch.ones_like(d[:,i:i+1]))
                    cond += [d[:, i:i+1] + trainNoise] # collect input steps
                conditioning = torch.concat(cond, dim=2) # combine along channel dimension
                data = d[:, prevSteps:prevSteps+1]

            noise, predictedNoise = self.modelDecoder(conditioning=conditioning, data=data)
            return noise, predictedNoise


        # INFERENCE
        else:
            prediction = torch.zeros_like(d, device="cuda" if self.useGPU else "cpu")

            if "+Enc" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                latentSpace = self.modelEncoder(d[:,0:1])
                l = torch.concat((latentSpace, simParams[:,0:1]), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])
                for i in range(1,sizeSeq):
                    cond = torch.concat((conditioning, prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            elif "+First" in self.p_md.arch:
                prediction[:,0] = d[:,0] # no prediction of first step
                for i in range(1,sizeSeq):
                    cond = torch.concat((d[:,0:1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result


            else:
                if "+Prev" in self.p_md.arch:
                    prevSteps = 2
                elif "+2Prev" in self.p_md.arch:
                    prevSteps = 3
                elif "+3Prev" in self.p_md.arch:
                    prevSteps = 4
                else:
                    prevSteps = 1

                #for i in range(4):
                for i in range(prevSteps): # no prediction of first steps
                    prediction[:,i] = d[:,i] 

                for i in range(prevSteps, sizeSeq):
                    cond = []
                    for j in range(prevSteps,0,-1):
                        cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
                    cond = torch.concat(cond, dim=2) # combine along channel dimension

                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            return prediction



    # Decoder diffusion model conditioned on latent space
    def forwardDiffusionDecode(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        # add simulation parameters to latent space
        l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace

        # match dimensionality
        cond = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

        if self.training:
            noise, predictedNoise = self.modelDecoder(conditioning=cond, data=d) # prediction conditioned on latent space
            return noise, predictedNoise
        else:
            prediction = self.modelDecoder(conditioning=cond, data=d)
            return prediction


    # Diffusion model conditioned on normal decoder ouput to finetune it
    def forwardDiffusionFinetune(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]

        cond, _ = self.modelDecoder[0](latentSpace, simParams)

        if self.training:
            noise, predictedNoise = self.modelDecoder[1](conditioning=cond, data=d)
            return noise, predictedNoise
        else:
            prediction = self.modelDecoder[1](conditioning=cond, data=d)
            return prediction


    # Diffusion model predicts next step based on previous step and secondary transformer network conditioning
    def forwardDiffusionHybrid(self, d:torch.Tensor, latentSpace:torch.Tensor, simParams:torch.Tensor) -> torch.Tensor:
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        if self.training:
            if "+Lat" in self.p_md.arch:
                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                conditioning = torch.concat((conditioning[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                noise, predictedNoise = self.modelDecoder(conditioning=conditioning, data=data)

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                randIndex = torch.randint(1, sizeSeq-1, (1,), device=d.device)

                conditioning = torch.concat((predictionAeDec[:,randIndex:randIndex+1], d[:,randIndex-1:randIndex]), dim=2)
                data = d[:,randIndex:randIndex+1]

                noise, predictedNoise = self.modelDecoder[1](conditioning=conditioning, data=data)

            return noise, predictedNoise

        else:
            prediction = torch.zeros_like(d, device="cuda" if self.useGPU else "cpu")
            prediction[:,0] = d[:,0] # no prediction of first and last step
            prediction[:,d.shape[1]-1] = d[:,d.shape[1]-1]

            if "+Lat" in self.p_md.arch:
                l = torch.concat((latentSpace, simParams), dim=2) if simParams is not None else latentSpace
                conditioning = l.unsqueeze(3).unsqueeze(4).expand(-1,-1,-1,d.shape[3],d.shape[4])

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((conditioning[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder(conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            elif "skip+" in self.p_md.arch:
                predictionAeDec, _ = self.modelDecoder[0](latentSpace, simParams)

                for i in range(1,sizeSeq-1):
                    cond = torch.concat((predictionAeDec[:,i:i+1], prediction[:,i-1:i]), dim=2)
                    result = self.modelDecoder[1](conditioning=cond, data=d[:,i-1:i])
                    if self.p_d.simParams:
                        result[:,:,-len(self.p_d.simParams):] = d[:,i:i+1,-len(self.p_d.simParams):] # replace simparam prediction with true values
                    prediction[:,i:i+1] = result

            return prediction


    def printModelInfo(self):
        pTrain = filter(lambda p: p.requires_grad, self.parameters())
        paramsTrain = sum([np.prod(p.size()) for p in pTrain])
        params = sum([np.prod(p.size()) for p in self.parameters()])

        if self.modelEncoder:
            pTrainEnc = filter(lambda p: p.requires_grad, self.modelEncoder.parameters())
            paramsTrainEnc = sum([np.prod(p.size()) for p in pTrainEnc])
            paramsEnc = sum([np.prod(p.size()) for p in self.modelEncoder.parameters()])
        if self.modelDecoder:
            pTrainDec = filter(lambda p: p.requires_grad, self.modelDecoder.parameters())
            paramsTrainDec = sum([np.prod(p.size()) for p in pTrainDec])
            paramsDec = sum([np.prod(p.size()) for p in self.modelDecoder.parameters()])
        if self.modelLatent:
            pTrainLat = filter(lambda p: p.requires_grad, self.modelLatent.parameters())
            paramsTrainLat = sum([np.prod(p.size()) for p in pTrainLat])
            paramsLat = sum([np.prod(p.size()) for p in self.modelLatent.parameters()])

        print("Weights Trainable (All): %d (%d)   %s   %s   %s" %
                (paramsTrain, params,
                ("Enc: %d (%d)" % (paramsTrainEnc, paramsEnc)) if self.modelEncoder else "",
                ("Dec: %d (%d)" % (paramsTrainDec, paramsDec)) if self.modelDecoder else "",
                ("Lat: %d (%d)" % (paramsTrainLat, paramsLat)) if self.modelLatent else ""))
        print(self)
        print("Data parameters: %s" % str(self.p_d.asDict()))
        print("Training parameters: %s" % str(self.p_t.asDict()))
        print("Loss parameters: %s" % str(self.p_l.asDict()))
        if self.p_me:
            print("Model Encoder parameters: %s" % str(self.p_me.asDict()))
        if self.p_md:
            print("Model Decoder parameters: %s" % str(self.p_md.asDict()))
        if self.p_ml:
            print("Model Latent parameters: %s" % str(self.p_ml.asDict()))
        print("")

        logging.info("Weights Trainable (All): %d (%d)   %s   %s   %s" %
                (paramsTrain, params,
                ("Enc: %d (%d)" % (paramsTrainEnc, paramsEnc)) if self.modelEncoder else "",
                ("Dec: %d (%d)" % (paramsTrainDec, paramsDec)) if self.modelDecoder else "",
                ("Lat: %d (%d)" % (paramsTrainLat, paramsLat)) if self.modelLatent else ""))
        logging.info(self)
        logging.info("Data parameters: %s" % str(self.p_d.asDict()))
        logging.info("Training parameters: %s" % str(self.p_t.asDict()))
        logging.info("Loss parameters: %s" % str(self.p_l.asDict()))
        if self.p_me:
            logging.info("Model Encoder parameters: %s" % str(self.p_me.asDict()))
        if self.p_md:
            logging.info("Model Decoder parameters: %s" % str(self.p_md.asDict()))
        if self.p_ml:
            logging.info("Model Latent parameters: %s" % str(self.p_ml.asDict()))
        logging.info("")



    @classmethod
    def load(cls, path:str, useGPU:bool=True):
        if useGPU:
            print('Loading model from %s' % path)
            loaded = torch.load(path)
        else:
            print('CPU - Loading model from %s' % path)
            loaded = torch.load(path, map_location=torch.device('cpu'))

        p_me = ModelParamsEncoder().fromDict(loaded['modelParamsEncoder']) if loaded['modelParamsEncoder'] else None
        p_md = ModelParamsDecoder().fromDict(loaded['modelParamsDecoder']) if loaded['modelParamsDecoder'] else None
        p_ml = ModelParamsLatent().fromDict(loaded['modelParamsLatent'])   if loaded['modelParamsLatent'] else None
        p_d = DataParams().fromDict(loaded['dataParams'])                  if loaded['dataParams'] else None
        p_t = TrainingParams().fromDict(loaded['trainingParams'])          if loaded['trainingParams'] else None
        p_l = LossParams().fromDict(loaded['lossParams'])                  if loaded['lossParams'] else None

        stateDictEncoder = loaded['stateDictEncoder']
        stateDictDecoder = loaded['stateDictDecoder']
        stateDictLatent = loaded['stateDictLatent']

        model = cls(p_d, p_t, p_l, p_me, p_md, p_ml, "", useGPU)

        if stateDictEncoder:
            model.modelEncoder.load_state_dict(stateDictEncoder)
        if stateDictDecoder:
            model.modelDecoder.load_state_dict(stateDictDecoder)
        if stateDictLatent:
            model.modelLatent.load_state_dict(stateDictLatent)
        model.eval()

        return model


    def save(self, basePath:str, epoch:int=-1, noPrint:bool=False):
        if not noPrint:
            print('Saving model to %s' % basePath)

        saveDict = {
            'stateDictEncoder'   : self.modelEncoder.state_dict() if self.modelEncoder else None,
            'stateDictDecoder'   : self.modelDecoder.state_dict() if self.modelDecoder else None,
            'stateDictLatent'    : self.modelLatent.state_dict() if self.modelLatent else None,
            'modelParamsEncoder' : self.p_me.asDict() if self.p_me else None,
            'modelParamsDecoder' : self.p_md.asDict() if self.p_md else None,
            'modelParamsLatent'  : self.p_ml.asDict() if self.p_ml else None,
            'dataParams'         : self.p_d.asDict() if self.p_d else None,
            'trainingParams'     : self.p_t.asDict() if self.p_t else None,
            'lossParams'         : self.p_l.asDict() if self.p_l else None,
            }

        if epoch > 0:
            path = os.path.join(basePath, "Model_E%03d.pth" % epoch)
        else:
            path = os.path.join(basePath, "Model.pth")
        torch.save(saveDict, path)

