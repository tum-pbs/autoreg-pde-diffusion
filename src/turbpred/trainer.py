import logging
import time, math

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from turbpred.model import PredictionModel
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.params import DataParams, TrainingParams


class Trainer(object):
    model: PredictionModel
    trainLoader: DataLoader
    optimizer: Optimizer
    lrScheduler: _LRScheduler
    criterion: PredictionLoss
    trainHistory: LossHistory
    writer: SummaryWriter
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, trainLoader:DataLoader, optimizer:Optimizer, lrScheduler:_LRScheduler,
            criterion:PredictionLoss, trainHistory:LossHistory, writer:SummaryWriter, p_d:DataParams, p_t:TrainingParams):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.lrScheduler = lrScheduler
        self.criterion = criterion
        self.trainHistory = trainHistory
        self.writer = writer
        self.p_d = p_d
        self.p_t = p_t

        self.seqenceLength = self.p_d.sequenceLength[0]

        if self.p_t.fadeInSeqLen[0] > 0 and not self.p_t.fadeInSeqLenLin:
            self.currentSeqLen = 2
            numSeqInc = math.ceil(math.log2(self.seqenceLength / 2.0))
            seqIncStep = math.floor((self.p_t.fadeInSeqLen[1] - self.p_t.fadeInSeqLen[0]) / (numSeqInc-1))
            self.seqIncreaseSteps = list(range(self.p_t.fadeInSeqLen[0], self.p_t.fadeInSeqLen[1]+1, seqIncStep))

            assert (numSeqInc == len(self.seqIncreaseSteps)), "Sequence length computation problem"
            print("Sequence length schedule: %d increases by factor 2 at epochs %s" % (numSeqInc, str(self.seqIncreaseSteps)))
            print("")
            logging.info("Sequence length schedule: %d increases by factor 2 at epochs %s" % (numSeqInc, str(self.seqIncreaseSteps)))
            logging.info("")

        elif self.p_t.fadeInSeqLen[0] > 0 and self.p_t.fadeInSeqLenLin:
            self.currentSeqLen = 2
            numSeqInc = max(0, self.seqenceLength - 2)
            seqIncStep = math.floor((self.p_t.fadeInSeqLen[1] - self.p_t.fadeInSeqLen[0]) / (numSeqInc-1))
            self.seqIncreaseSteps = list(range(self.p_t.fadeInSeqLen[0], self.p_t.fadeInSeqLen[1]+1, seqIncStep))

            seqIncStr = str(self.seqIncreaseSteps) if len(self.seqIncreaseSteps) < 10 else str(self.seqIncreaseSteps[:4]) + " ... " + str(self.seqIncreaseSteps[-4:])
            print("Sequence length schedule: %d increases by value 1 at epochs %s" % (numSeqInc, seqIncStr))
            print("")
            logging.info("Sequence length schedule: %d increases by value 1 at epochs %s" % (numSeqInc, seqIncStr))
            logging.info("")

        else:
            self.currentSeqLen = self.seqenceLength
            self.seqIncreaseSteps = []


    # run one epoch of training
    def trainingStep(self, epoch:int):
        assert (len(self.trainLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        if self.currentSeqLen < self.seqenceLength and epoch in self.seqIncreaseSteps:
            if not self.p_t.fadeInSeqLenLin:
                self.currentSeqLen = min(2 * self.currentSeqLen, self.seqenceLength)
            else:
                self.currentSeqLen = min(1 + self.currentSeqLen, self.seqenceLength)

        self.model.train()
        for s, sample in enumerate(self.trainLoader, 0):
            self.optimizer.zero_grad()

            device = "cuda" if self.model.useGPU else "cpu"
            data = sample["data"].to(device)
            simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
            if "obsMask" in sample:
                obsMask = sample["obsMask"].to(device)
                obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
            else:
                obsMask = None

            fadePredStart = self.p_t.fadeInPredLoss[0]
            fadePredEnd = self.p_t.fadeInPredLoss[1]
            fade = (epoch - fadePredStart) / (fadePredEnd - fadePredStart)
            fade = max(min(fade, 1), 0)

            # train prediction samples with AE until fading starts and latent network becomes active
            fadeWeight = fade if fade > 0 else 1
            useLatent = fade > 0

            prediction, latentSpace, vaeMeanVar = self.model(data, simParameters, useLatent=useLatent)

            if self.currentSeqLen < self.seqenceLength:
                if not vaeMeanVar[0] is None and not vaeMeanVar[1] is None:
                    vaeMeanVar = (vaeMeanVar[0][:,0:self.currentSeqLen], vaeMeanVar[1][:,0:self.currentSeqLen])

                p = prediction[:,0:self.currentSeqLen]
                d = data[:,0:self.currentSeqLen]
                l = latentSpace[:,0:self.currentSeqLen]
            else:
                p = prediction
                d = data
                l = latentSpace

            #if obsMask is not None:
            #    p = p * obsMask
            #    d = d * obsMask

            ignorePredLSIMSteps = 0
            # ignore loss on scalar simulation parameters that are replaced in unet rollout during training
            if self.model.p_md.arch in ["unet", "unet+Prev", "unet+2Prev", "unet+3Prev",
                                    "dil_resnet", "dil_resnet+Prev", "dil_resnet+2Prev", "dil_resnet+3Prev",
                                    "resnet", "resnet+Prev", "resnet+2Prev", "resnet+3Prev",
                                    "fno", "fno+Prev", "fno+2Prev", "fno+3Prev",
                                    "dfp", "dfp+Prev", "dfp+2Prev", "dfp+3Prev",]:
                numFields = self.p_d.dimension + len(self.p_d.simFields)
                p = p[:,:,0:numFields]
                d = d[:,:,0:numFields]
                if "+Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 1
                elif "+2Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 2
                elif "+3Prev" in self.model.p_md.arch:
                    ignorePredLSIMSteps = 3

            loss, lossParts, lossSeq = self.criterion(p, d, l, vaeMeanVar, fadePredWeight=fadeWeight, ignorePredLSIMSteps=ignorePredLSIMSteps)

            loss.backward()

            self.optimizer.step()

            timerEnd = time.perf_counter()
            self.trainHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

        self.lrScheduler.step()

        timerEnd = time.perf_counter()
        self.trainHistory.updateEpoch((timerEnd-timerStart)/60.0)

        if epoch % 50 == 49:
            self.model.eval()
            with torch.no_grad():
                if obsMask is not None:
                    maskedPred = prediction * obsMask
                    maskedData = data * obsMask
                else:
                    maskedPred = prediction
                    maskedData = data

                self.trainHistory.writePredictionExample(maskedPred, maskedData)
                self.trainHistory.writeSequenceLoss(lossSeq)

        self.trainHistory.prepareAndClearForNextEpoch()




class Tester(object):
    model: PredictionModel
    testLoader: DataLoader
    criterion: PredictionLoss
    testHistory: LossHistory
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, testLoader:DataLoader, criterion:PredictionLoss,
                    testHistory:LossHistory, p_t:TrainingParams):
        self.model = model
        self.testLoader = testLoader
        self.criterion = criterion
        self.testHistory = testHistory
        self.p_t = p_t


    # run one epoch of testing
    def testStep(self, epoch:int):
        if epoch % self.testHistory.epochStep != self.testHistory.epochStep - 1:
            return

        assert (len(self.testLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        self.model.eval()
        with torch.no_grad():
            for s, sample in enumerate(self.testLoader, 0):
                device = "cuda" if self.model.useGPU else "cpu"
                data = sample["data"].to(device)
                simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
                if "obsMask" in sample is not None:
                    obsMask = sample["obsMask"].to(device)
                    obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
                else:
                    obsMask = None

                prediction, latentSpace, vaeMeanVar = self.model(data, simParameters, useLatent=True)

                p = prediction
                d = data
                l = latentSpace

                if obsMask is not None:
                    p = p * obsMask
                    d = d * obsMask

                _, lossParts, lossSeq = self.criterion(p, d, l, vaeMeanVar, weighted=False)

                timerEnd = time.perf_counter()
                self.testHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

            timerEnd = time.perf_counter()
            self.testHistory.updateEpoch((timerEnd-timerStart)/60.0)

            #if epoch % 50 == 49:
            if obsMask is not None:
                maskedPred = prediction * obsMask
                maskedData = data * obsMask
            else:
                maskedPred = prediction
                maskedData = data

            self.testHistory.writePredictionExample(maskedPred, maskedData)
            self.testHistory.writeSequenceLoss(lossSeq)

            self.testHistory.prepareAndClearForNextEpoch()

