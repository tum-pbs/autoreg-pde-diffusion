import torch
from torch.utils.data import DataLoader

from turbpred.turbulence_dataset import TurbulenceDataset

combinedVelMean = True
comVelMeanWeights = [2,1]
dataset = TurbulenceDataset("All", ["data"], filterTop=["256_inc"], filterSim=[(0,91)], filterFrame=[(300,1300)],
            sequenceLength=[(1,1)], simFields=["pres"], simParams=["rey"], printLevel="sim")
#dataset = TurbulenceDataset("All", ["data"], filterTop=["256_tra"], filterSim=[(0,41)], filterFrame=[(0,1000)],
#            sequenceLength=[(1,1)], simFields=["pres","dens"], simParams=["mach"], printLevel="sim")
#dataset = TurbulenceDataset("All", ["data"], filterTop=["128_iso"], filterSim=[(0,1000)], filterFrame=[(0,1000)],
#            sequenceLength=[(1,1)], simFields=["velZ", "pres"], simParams=[], printLevel="sim")
dataset.transform = lambda x : x

dataLoader = DataLoader(dataset, batch_size=1024, drop_last=False, num_workers=8)

summed, count = 0, 0
for s, sample in enumerate(dataLoader, 0):
    print("Pass 1:  %d / %d" % (s, len(dataLoader)))
    # batch, seq=1, channel, x-dim, y-dim, (z-dim)
    data = sample["data"][:,0]
    summed += torch.sum(data, dim=(0,2,3))
    count += data.shape[0] * data.shape[2] * data.shape[3]
mean = summed / count

var, count = 0, 0
for s, sample in enumerate(dataLoader, 0):
    print("Pass 2:  %d / %d" % (s, len(dataLoader)))
    # batch, seq=1, channel, x-dim, y-dim, (z-dim)
    data = sample["data"][:,0]
    var += torch.sum((data - torch.reshape(mean, (1,-1,1,1))) ** 2, dim=(0,2,3))
    count += data.shape[0] * data.shape[2] * data.shape[3]

std = torch.sqrt(var / count)
print()
meanStr = ""
stdStr = ""
for i in range(2, mean.shape[0]):
    meanStr += "%1.6f, " % mean[i]
    stdStr += "%1.6f, " % std[i]
meanStr = meanStr[:len(meanStr)-2]
stdStr = stdStr[:len(stdStr)-2]
print("Means (vel_x, vel_y, pres, rey):")
#print("Means (vel_x, vel_y, dens, pres, mach):")
#print("Means (vel_x, vel_y, vel_z, pres):")
print("\t%1.6f, %1.6f, %s" % (mean[0], mean[1], meanStr))
print("Stds:")
print("\t%1.6f, %1.6f, %s" % (std[0], std[1], stdStr))

if combinedVelMean:
    comVelMean = (comVelMeanWeights[0] * mean[0] + comVelMeanWeights[1] * mean[1]) / sum(comVelMeanWeights)
    comVelStd = (comVelMeanWeights[0] * std[0] + comVelMeanWeights[1] * std[1]) / sum(comVelMeanWeights)
    print("\nMeans (combined velocity):")
    print("\t%1.6f, %1.6f, %s" % (comVelMean, comVelMean, meanStr))
    print("Stds (combined velocity):")
    print("\t%1.6f, %1.6f, %s" % (comVelStd, comVelStd, stdStr))


