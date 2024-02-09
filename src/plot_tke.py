import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm


plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42




datasetName = "zInterp"
modelMinMax = (0,3)
evalMinMax = (0,5)
sequenceMinMax = (10,11)
timeMinMax = (0,100)

predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"

models = {
    "Simulation": "groundTruth.dict",

    #"ResNet": "resnet-m2.npz",
    "Dil-ResNet": "dil-resnet-m2.npz",

    #"FNO16": "fno-16modes-m2.npz",
    "FNO32": "fno-32modes-m2.npz",

    #"TF-MGN": "tf-mgn.npz",
    "TF-Enc": "tf-enc.npz",
    #"TF-VAE": "tf-vae.npz",

    "U-Net": "unet-m2.npz",
    "U-Net-ut": "unet-m8.npz",
    "U-Net-tn": "unet-m2-noise0.01.npz",

    "Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),

    #"ACDM-ncn": "acdm100_ncn.npz",
    "ACDM": "acdm100.npz",
}

tkeAverageMinMax = (0,100)
useStd = True

modelNames = []
tkeMean = []
tkeStd = []
tkeQuantileLower = []
tkeQuantileUpper = []

for modelName, modelPath in models.items():
    modelNames += [modelName]

    if modelPath == "groundTruth.dict":
        groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
        groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
        if "obsMask" in groundTruthDict:
            obsMask = groundTruthDict["obsMask"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
            groundTruth = groundTruth * obsMask # ignore obstacle area
        print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
        prediction = groundTruth[:,:,
                                sequenceMinMax[0]:sequenceMinMax[1],
                                timeMinMax[0]:timeMinMax[1],
                                getLossRelevantFields(datasetName)[0]:getLossRelevantFields(datasetName)[1]]
        print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

    else:
        fullPath = os.path.join(predictionFolder, modelPath)
        prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
        if "obsMask" in groundTruthDict:
            prediction = prediction * obsMask
        prediction = prediction[modelMinMax[0]:modelMinMax[1],
                            evalMinMax[0]:evalMinMax[1],
                            sequenceMinMax[0]:sequenceMinMax[1],
                            timeMinMax[0]:timeMinMax[1],
                            getLossRelevantFields(datasetName)[0]:getLossRelevantFields(datasetName)[1]]
        print("Loaded prediction from model %s with shape: %s" % (modelName, str(list(prediction.shape))))

    vel = prediction[:,:,:,:, 0:2]
    velFluc = vel - torch.mean(vel, dim=3, keepdim=True)

    fftX = torch.fft.fft(vel, dim=5)
    fftY = torch.fft.fft(vel, dim=6)

    n = min(fftX.shape[5], fftY.shape[6]) # minimal number of frequencies in fourier space
    if datasetName in ["extrap", "interp", "longer"]:
        gridSpacing = (6.0/fftY.shape[6]) # spacing determined by 12x6 interpolation area
    elif datasetName in ["zInterp"]:
        gridSpacing = (2 * ((2*3.1415) / 1024)) # spacing determined by 2pi x 2pi full simulation area of resolution 1024x1024 with strided queries of 2
    freq = np.fft.fftfreq(n, d=gridSpacing)[1:int(n/2)]

    energyX = torch.real( torch.sum(fftX * torch.conj(fftX), dim=4, keepdim=True) )
    energyY = torch.real( torch.sum(fftY * torch.conj(fftY), dim=4, keepdim=True) )
    energyX = torch.mean( energyX[:,:,:,:,:,1:int(n/2)],   dim=6) # only use positive fourier frequencies
    energyY = torch.mean( energyY[:,:,:,:,:,:,1:int(n/2)], dim=5)

    energy = torch.squeeze(0.5 * (energyX + energyY), dim=4)
    energy = torch.mean(energy[:,:,:,tkeAverageMinMax[0]:tkeAverageMinMax[1]], dim=3)
    tkeMean += [torch.mean(energy, dim=(0,1,2)).numpy()]
    tkeStd += [torch.std(energy, dim=(0,1,2)).numpy()]
    tkeQuantileLower += [np.quantile(energy.numpy(), 0.05, axis=(0,1,2))]
    tkeQuantileUpper += [np.quantile(energy.numpy(), 0.95, axis=(0,1,2))]


fig, ax = plt.subplots(1, figsize=(5.0,2.3), dpi=150)
ax.text(0.008, 0.018, getDatasetName(datasetName), color="k", bbox=dict(facecolor="whitesmoke", edgecolor="darkslategray", boxstyle="round"),
        horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

ax.set_xlabel("Wavenumber $\kappa$ (temporal average)")
ax.set_ylabel("TKE $*\kappa^2$")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
#ax.set_ylim([0,5000])
ax.yaxis.grid(True)
ax.set_axisbelow(True)
for i in range(len(modelNames)):
    tkeMean[i] = tkeMean[i] * freq * freq
    tkeStd[i] = tkeStd[i] * freq * freq
    tkeQuantileLower[i] = tkeQuantileLower[i] * freq * freq
    tkeQuantileUpper[i] = tkeQuantileUpper[i] * freq * freq
    color = getColor(modelNames[i])
    label = getModelName(modelNames[i])
    if modelNames[i] == "Simulation":
        ms = np.logspace(5, 0.1, freq.shape[0], base=2)
        ax.plot(freq, tkeMean[i], linewidth=1.5, color=color, linestyle="dotted")
        ax.scatter(freq, tkeMean[i], ms, color=color, marker="o")
        ax.plot([], [], linewidth=1.5, color=color, label=label, linestyle="dotted", marker="o", markersize=4)
    else:
        ls = "dashdot" if "Pre" in modelNames[i] else "solid"
        ax.plot(freq, tkeMean[i], linewidth=1.5, color=color, label=label, linestyle=ls)
        if useStd:
            #ax.fill_between(freq, tkeMean[i] - tkeStd[i], tkeMean[i] + tkeStd[i], facecolor=color, alpha=0.15)
            ax.fill_between(freq, tkeQuantileLower[i], tkeQuantileUpper[i], facecolor=color, alpha=0.15)
#ax.legend(bbox_to_anchor=(1.01, 1.15))

fig.tight_layout(pad=0.4)
fig.savefig("%s/tke_%s.pdf" % (outputFolder, datasetName))




