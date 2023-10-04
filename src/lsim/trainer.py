import torch
from lsim.distance_model import *
from lsim.base_models import *
from lsim.loss import *
import time


class Trainer(object):
    def __init__(self, model, trainLoader, optimizer, criterion, printEvery, showProgressPrint=False):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.criterion = criterion
        self.printEvery = printEvery
        self.showProgressPrint = showProgressPrint

        self.histLoss = []
        self.histLossL2 = []
        self.histLossCor = []
        self.histCor = []

    def trainingStep(self, epoch):
        start = time.time()

        runningLoss = 0
        runningLossL2 = 0
        runningLossCor = 0
        runningCor = 0

        for i, sample in enumerate(self.trainLoader, 0):
            self.optimizer.zero_grad()

            output = self.model(sample)
            actual = sample["distance"].cuda()

            loss = self.criterion(output, actual)
            loss.backward()
            self.optimizer.step()

            runningLoss += loss.item()
            runningLossL2 += self.criterion.lastForwardLossL2
            runningLossCor += self.criterion.lastForwardLossCor
            runningCor += self.criterion.lastForwardCor

            del loss

            if i%6 == 0 and self.showProgressPrint:
                print("Training Epoch:", round((100*i)/len(self.trainLoader), 1), "%", end="\r")

            if i % self.printEvery == self.printEvery - 1:
                count = self.printEvery * sample["reference"].shape[0]
                runningLoss = runningLoss / count
                runningLossL2 = runningLossL2 / count
                runningLossCor = runningLossCor / count
                runningCor = runningCor / count

                self.histLoss.append(runningLoss)
                self.histLossL2.append(runningLossL2)
                self.histLossCor.append(runningLossCor)
                self.histCor.append(runningCor)

                end = time.time()
                print('%3ds [%2d, %4d] loss: %.5f(%.3f + %.3f) cor: %.3f' % (end-start, epoch, i + 1, runningLoss,
                                                        runningLossL2, runningLossCor, runningCor))
                runningLoss = 0
                runningLossL2 = 0
                runningLossCor = 0
                runningCor = 0

        if self.showProgressPrint:
            print("                                                          ", end="\r") #override temporary percentage print

    def normCalibration(self, epochs, stopEarly=0):
        if self.model.normMode == "normUnit":
            return

        self.model.eval()
        with torch.no_grad():
            for epoch in range(epochs):
                for i, sample in enumerate(self.trainLoader, 0):
                    if stopEarly > 0 and i >= stopEarly:
                        break
                    if i%8 == 0 and self.showProgressPrint:
                        print("Norm calibration:", round((100*(i+epoch*len(self.trainLoader)))/(epochs*len(self.trainLoader)), 1), "%", end="\r")

                    self.model.updateNorm(sample)

        self.model.train()
        print("Norm calibration: completed")


    def save(self, path):
        #print('Saving trainer to %s' % path)
        histDict = {'histLoss' : self.histLoss,
                    'histLossL2' : self.histLossL2,
                    'histLossCor': self.histLossCor,
                    'histCor' : self.histCor, }
        torch.save(histDict, path)

    def load(self, path):
        #print('Loading trainer from %s' % path)
        histDict = torch.load(path)
        self.histLoss = histDict['histLoss']
        self.histLossL2 = histDict['histLossL2']
        self.histLossCor = histDict['histLossCor']
        self.histCor = histDict['histCor']


class Validator(object):
    def __init__(self, model, valLoader, criterion):
        self.model = model
        self.valLoader = valLoader
        self.criterion = criterion

        self.histDistMean = []
        self.histDistStd = []
        self.histCorMean = []
        self.histCorStd = []
        self.histCorFull = []

    def validationStep(self):
        start = time.time()

        distance = []
        correlation = []
        output = []
        actual = []

        with torch.no_grad():
            start = time.time()

            self.model.eval()
            self.model.isTrain = False

            for i, sample in enumerate(self.valLoader, 0):
                out = self.model(sample)
                act = sample["distance"].cuda()

                output.append(out[:10].cpu().numpy())
                actual.append(act[:10].cpu().numpy())

                dist = self.criterion.distanceL2(out, act).detach()
                corSingle = self.criterion.corrcoef(out[:10], act[:10])

                distance.append(dist.cpu().numpy())
                correlation.append(corSingle.cpu().numpy())
                del dist
                del corSingle

            self.model.train()
            self.model.isTrain = True

            distance = np.vstack(distance)
            distMean = distance.mean()
            distStd = distance.std()
            self.histDistMean.append(distMean)
            self.histDistStd.append(distStd)

            correlation = np.vstack(correlation)
            corMean = correlation.mean()
            corStd = correlation.std()
            self.histCorMean.append(corMean)
            self.histCorStd.append(corStd)

            self.histCorFull.append(0)

            end = time.time()
            print("%ds Validation (distance error mean/std) (correlation mean/std):" % (end-start))
            print( "%1.3f (%1.3f) -- %1.4f (%1.4f)" % (distMean, distStd, corMean, corStd) )
            print()

    def save(self, path):
        #print('Saving validator to %s' % path)
        histDict = {'histDistMean' : self.histDistMean,
                    'histDistStd' : self.histDistStd,
                    'histCorMean': self.histCorMean,
                    'histCorStd': self.histCorStd,
                    'histCorFull' : self.histCorFull, }
        torch.save(histDict, path)

    def load(self, path):
        #print('Loading validator from %s' % path)
        histDict = torch.load(path)
        self.histDistMean = histDict['histDistMean']
        self.histDistStd = histDict['histDistStd']
        self.histCorMean = histDict['histCorMean']
        self.histCorStd = histDict['histCorStd']
        self.histCorFull = histDict['histCorFull']
