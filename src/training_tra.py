import os
import copy
from typing import Dict
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler

from turbpred.model import PredictionModel
from turbpred.logger import Logger
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.trainer import Trainer, Tester


if __name__ == "__main__":
    useGPU = True
    gpuID = "0"

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)

    ### UNET
    modelName = "2D_Tra/128_unet-m2"
    p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
                dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    p_t = TrainingParams(epochs=1000, lr=0.0001)
    p_l = LossParams(recMSE=0.0, predMSE=1.0)
    p_me = None
    p_md = ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.0)
    p_ml = None
    pretrainPath = ""

    ### UNET_ut
    # modelName = "2D_Tra/128_unet-m8"
    # p_d = DataParams(batch=16, augmentations=["normalize"], sequenceLength=[8,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=1000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="unet", pretrained=False)
    # p_ml = None
    # pretrainPath = ""

    ### UNET_tn
    # modelName = "2D_Tra/128_unet-m2-noise0.01"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=1000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.01)
    # p_ml = None
    # pretrainPath = ""


    ### RESNET
    # modelName = "2D_Tra/128_resnet-m2"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=1000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="resnet", pretrained=False, trainingNoise=0.0, decWidth=144)
    # p_ml = None
    # pretrainPath = ""

    ### RESNET_dil.
    # modelName = "2D_Tra/128_dil-resnet-m2"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=1000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="dil_resnet", pretrained=False, trainingNoise=0.0, decWidth=144)
    # p_ml = None
    # pretrainPath = ""


    ### FNO_16
    # modelName = "2D_Tra/128_fno-16modes-m2"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=2000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="fno", pretrained=False, trainingNoise=0.0, decWidth=112, fnoModes=(16,8))
    # p_ml = None
    # pretrainPath = ""

    ### FNO_32
    # modelName = "2D_Tra/128_fno-32modes-m2"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[2,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=2000, lr=0.0001)
    # p_l = LossParams(recMSE=0.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = None
    # p_md = ModelParamsDecoder(arch="fno", pretrained=False, trainingNoise=0.0, decWidth=56, fnoModes=(32,16))
    # p_ml = None
    # pretrainPath = ""


    ### TF_Enc
    # modelName = "2D_Tra/128_tf-enc"
    # p_d = DataParams(batch=8, augmentations=["normalize"], sequenceLength=[60,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=5000, lr=0.0001, fadeInSeqLen=[300,1200], fadeInSeqLenLin=True)
    # p_l = LossParams(recMSE=1.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = ModelParamsEncoder(arch="skip", pretrained=False, encWidth=32, latentSize=31)
    # p_md = ModelParamsDecoder(arch="skip", pretrained=False, decWidth=96, vae=False)
    # p_ml = ModelParamsLatent(arch="transformerEnc", pretrained=False, width=1024, layers=1, dropout=0.0,
    #             transTrainUnroll=True, transTargetFull=True, maxInputLen=30)
    # pretrainPath = ""

    ### TF_VAE
    # modelName = "2D_Tra/128_tf-vae"
    # p_d = DataParams(batch=8, augmentations=["normalize"], sequenceLength=[60,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=5000, lr=0.0001, fadeInSeqLen=[300,1200], fadeInSeqLenLin=True)
    # p_l = LossParams(recMSE=1.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0, regVae=0.1)
    # p_me = ModelParamsEncoder(arch="skip", pretrained=False, encWidth=32, latentSize=63)
    # p_md = ModelParamsDecoder(arch="skip", pretrained=False, decWidth=96, vae=True)
    # p_ml = ModelParamsLatent(arch="transformerEnc", pretrained=False, width=1024, layers=1, dropout=0.0,
    #             transTrainUnroll=True, transTargetFull=True, maxInputLen=30)
    # pretrainPath = ""

    ### TF_MGN
    # modelName = "2D_Tra/128_tf-mgn"
    # p_d = DataParams(batch=8, augmentations=["normalize"], sequenceLength=[60,2], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")
    # p_t = TrainingParams(epochs=5000, lr=0.0001, fadeInSeqLen=[300,1200], fadeInSeqLenLin=True)
    # p_l = LossParams(recMSE=1.0, recLSIM=0.0, predMSE=1.0, predLSIM=0.0)
    # p_me = ModelParamsEncoder(arch="skip", pretrained=False, encWidth=32, latentSize=32)
    # p_md = ModelParamsDecoder(arch="skip", pretrained=False, decWidth=96, vae=False)
    # p_ml = ModelParamsLatent(arch="transformerMGN", pretrained=False, width=1024, layers=1, dropout=0.0,
    #             transTrainUnroll=True, transTargetFull=False, maxInputLen=30)
    # pretrainPath = ""




    trainSet = TurbulenceDataset("Training", ["data"], filterTop=["128_tra"], filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, filterFrame=[(0,1000)],
                    sequenceLength=[p_d.sequenceLength], randSeqOffset=p_d.randSeqOffset, simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim")

    testSets = {
        "extrap":
            TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", ["data"], filterTop=["128_tra"], filterSim=[(0,3)],
                    filterFrame=[(500,750)], sequenceLength=[[60,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
        "interp" :
            TurbulenceDataset("Test Interpolate Mach 0.66-0.68", ["data"], filterTop=["128_tra"], filterSim=[(16,19)],
                    filterFrame=[(500,750)], sequenceLength=[[60,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
        "longer" :
            TurbulenceDataset("Test Longer Rollout Mach 0.64-0.65", ["data"], filterTop=["128_tra"], filterSim=[(14,16)],
                    filterFrame=[(0,1000)], sequenceLength=[[240,2]], simFields=p_d.simFields, simParams=p_d.simParams, printLevel="sim"),
    }



def train(modelName:str, trainSet:TurbulenceDataset, testSets:Dict[str,TurbulenceDataset],
        p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
        p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True, gpuID:str="0"):

    # DATA AND MODEL SETUP
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    logger = Logger(modelName, addNumber=True)
    model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    model.printModelInfo()
    criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_t.lr, weight_decay=p_t.weightDecay)
    lrScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p_t.expLrGamma)
    logger.setup(model, optimizer)

    transTrain = Transforms(p_d)
    trainSet.transform = transTrain
    trainSet.printDatasetInfo()
    trainSampler = RandomSampler(trainSet)
    #trainSampler = SubsetRandomSampler(range(2))
    trainLoader = DataLoader(trainSet, sampler=trainSampler,
                    batch_size=p_d.batch, drop_last=True, num_workers=4)
    trainHistory = LossHistory("_train", "Training", logger.tfWriter, len(trainLoader),
                                    0, 1, printInterval=1, logInterval=1, simFields=p_d.simFields)
    trainer = Trainer(model, trainLoader, optimizer, lrScheduler, criterion, trainHistory, logger.tfWriter, p_d, p_t)

    testers = []
    testHistories = []
    for shortName, testSet in testSets.items():
        p_d_test = copy.deepcopy(p_d)
        p_d_test.augmentations = ["normalize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        if p_d.sequenceLength[0] != p_d_test.sequenceLength[0]:
            p_d_test.batch = 1

        transTest = Transforms(p_d_test)
        testSet.transform = transTest
        testSet.printDatasetInfo()
        testSampler = SequentialSampler(testSet)
        #testSampler = SubsetRandomSampler(range(2))
        testLoader = DataLoader(testSet, sampler=testSampler,
                        batch_size=p_d_test.batch, drop_last=False, num_workers=4)
        testHistory = LossHistory(shortName, testSet.name, logger.tfWriter, len(testLoader),
                                    100, 100, printInterval=0, logInterval=0, simFields=p_d.simFields)
        tester = Tester(model, testLoader, criterion, testHistory, p_t)
        testers += [tester]
        testHistories += [testHistory]

    #if loadEpoch > 0:
    #    logger.resumeTrainState(loadEpoch)

    # TRAINING
    print('Starting Training')
    logger.saveTrainState(0)

    for tester in testers:
        tester.testStep(0)

    for epoch in range(0, p_t.epochs):
        trainer.trainingStep(epoch)
        logger.saveTrainState(epoch)

        for tester in testers:
            tester.testStep(epoch+1)

        trainHistory.updateAccuracy([p_d,p_t,p_l,p_me,p_md,p_ml], testHistories, epoch==p_t.epochs-1)

    logger.close()

    print('Finished Training')


if __name__ == "__main__":
    train(modelName, trainSet, testSets, p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath=pretrainPath, useGPU=useGPU, gpuID=gpuID) #type:ignore