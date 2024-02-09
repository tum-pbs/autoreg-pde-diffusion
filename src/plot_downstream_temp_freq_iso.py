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

    "ACDM-ncn": "acdm100_ncn.npz",
    "ACDM": "acdm100.npz",
}

cylsDownstream = 1.0
cylsY = 0.0
field = "velX"
useStd = True


modelNames = []
lineMean = []
lineStd = []
lineQuantileLower = []
lineQuantileUpper = []

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

    if datasetName in ["extrap", "interp", "longer"]:
        xPos = int(((cylsDownstream + 2.0)/12.0) * prediction.shape[5])
        yPos = int((cylsY/3.0) * 0.5 * prediction.shape[6] + 0.5 * prediction.shape[6])
    elif datasetName in ["highRey", "lowRey", "varReyIn"]:
        xPos = int(((0.6 * cylsDownstream + 1.3)/4.0) * prediction.shape[5])
        yPos = int((0.6 * cylsY) * 0.5 * prediction.shape[6] + 0.5 * prediction.shape[6])
    elif datasetName in ["zInterp"]:
        xPos = int(cylsDownstream * prediction.shape[5])
        yPos = int(cylsY * prediction.shape[6])
    else:
        raise ValueError("Problem with line position computation, invalid dataset!")

    if datasetName in ["extrap", "interp", "longer", "highRey", "lowRey", "varReyIn"]:
        pointField = prediction[:,:,:,:, getFieldIndex(datasetName, field):getFieldIndex(datasetName, field)+1, xPos:xPos+1, yPos:yPos+1]
    elif datasetName in ["zInterp"]:
        pointField = prediction[:,:,:,:, getFieldIndex(datasetName, field):getFieldIndex(datasetName, field)+1, :, :]

    fft = torch.fft.fft(pointField, dim=3)
    fft = torch.real(fft * torch.conj(fft))
    n = fft.shape[3]
    gridSpacing = 0.002 if datasetName in ["zInterp"] else 1 # delta t between frames as 1 here
    freq = np.fft.fftfreq(n, d=gridSpacing)[1:int(n/2)]
    fft = fft[:,:,:,1:int(n/2),:,:,:] # only use positive fourier frequencies

    fftMean = torch.mean(fft, dim=(5,6))
    lineMean += [torch.mean(fftMean, dim=(0,1,2,4)).numpy()]
    lineStd += [torch.std(fftMean, dim=(0,1,2,4)).numpy()]
    lineQuantileLower += [np.quantile(fftMean.numpy(), 0.05, axis=(0,1,2,4))]
    lineQuantileUpper += [np.quantile(fftMean.numpy(), 0.95, axis=(0,1,2,4))]



fig, ax = plt.subplots(1, figsize=(5.0,2.3), dpi=150)
ax.text(0.008, 0.018, getDatasetName(datasetName), color="k", bbox=dict(facecolor="whitesmoke", edgecolor="darkslategray", boxstyle="round"),
        horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
if datasetName in ["extrap", "interp", "longer", "highRey", "lowRey", "varReyIn"]:
    ax.set_xlabel("Temporal frequency (spatial average)")
elif datasetName in ["zInterp"]:
    ax.set_xlabel("Temporal frequency $f$ (spatial average)")
ax.set_ylabel("$v_x$ Amplitude $*f^2$")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
#ax.set_ylim([10**2,2*10**4])
for i in range(len(modelNames)):
    lineMean[i] = lineMean[i] * freq * freq
    lineStd[i] = lineStd[i] * freq * freq
    lineQuantileLower[i] = lineQuantileLower[i] * freq * freq
    lineQuantileUpper[i] = lineQuantileUpper[i] * freq * freq
    color = getColor(modelNames[i])
    label = getModelName(modelNames[i])
    if modelNames[i] == "Simulation":
        ms = np.logspace(5, 0.1, freq.shape[0], base=2)
        ax.plot(freq, lineMean[i], linewidth=1.5, color=color, linestyle="dotted")
        ax.scatter(freq, lineMean[i], ms, color=color, marker="o")
        ax.plot([], [], linewidth=1.5, color=color, label=label, linestyle="dotted", marker="o", markersize=4)
    else:
        ls = "dashdot" if "Pre" in modelNames[i] else "solid"
        ax.plot(freq, lineMean[i], linewidth=1.5, color=color, label=label, linestyle=ls)
        if useStd:
            #ax.fill_between(freq, lineMean[i] - lineStd[i], lineMean[i] + lineStd[i], facecolor=color, alpha=0.15)
            ax.fill_between(freq, lineQuantileLower[i], lineQuantileUpper[i], facecolor=color, alpha=0.15)
#ax.legend(bbox_to_anchor=(1.01, 1.06))

fig.tight_layout(pad=0.4)
fig.savefig("%s/downstream_temp_freq_%s_%s.pdf" % (outputFolder, datasetName, field))




