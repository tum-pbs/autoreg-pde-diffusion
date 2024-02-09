import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42




datasetName = "longer"
modelMinMax = (0,3)
evalMinMax = (0,5)
sequenceMinMax = (1,2)
timeMinMax = (0,240)

predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"

models = {
    "Simulation": "groundTruth.dict",

    #"ResNet": "resnet-m2.npz",
    #"Dil-ResNet": "dil-resnet-m2.npz",

    #"FNO16": "fno-16modes-m2.npz",
    #"FNO32": "fno-32modes-m2.npz",

    #"TF-MGN": "tf-mgn.npz",
    #"TF-Enc": "tf-enc.npz",
    #"TF-VAE": "tf-vae.npz",

    #"U-Net": "unet-m2.npz",
    #"U-Net-ut": "unet-m8.npz",
    #"U-Net-tn": "unet-m2-noise0.01.npz",

    #"Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),

    #"ACDM-ncn": "acdm20_ncn.npz",
    "ACDM": "acdm20.npz",
}

cylsDownstream = 1.0
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
        linePos = int(((cylsDownstream + 2.0)/12.0) * prediction.shape[5])
    elif datasetName in ["highRey", "lowRey", "varReyIn"]:
        linePos = int(((0.6 * cylsDownstream + 1.3)/4.0) * prediction.shape[5])
    elif datasetName in ["zInterp"]:
        linePos = int(cylsDownstream * prediction.shape[5])
    else:
        raise ValueError("Problem with line position computation, invalid dataset!")
    lineField = prediction[:,:,:,:, getFieldIndex(datasetName, field):getFieldIndex(datasetName, field)+1, linePos:linePos+1]
    meanLineField = torch.mean(lineField, dim=3, keepdim=True)

    fft = torch.fft.fft(meanLineField, dim=6)
    fft = torch.real(fft * torch.conj(fft))
    n = fft.shape[6]
    if datasetName in ["extrap", "interp", "longer"]:
        gridSpacing = (6.0/fft.shape[6]) # spacing determined by 12x6 interpolation area
    elif datasetName in ["zInterp"]:
        gridSpacing = (2 * ((2*3.1415) / 1024)) # spacing determined by 2pi x 2pi full simulation area of resolution 1024x1024 with strided queries of 2
    else:
        gridSpacing = 1
    freq = np.fft.fftfreq(n, d=gridSpacing)[1:int(n/2)]

    fft = fft[:,:,:,:,:,:,1:int(n/2)] # only use positive fourier frequencies

    lineMean += [torch.mean(fft, dim=(0,1,2,3,4,5)).numpy()]
    lineStd += [torch.std(fft, dim=(0,1,2,3,4,5)).numpy()]
    lineQuantileLower += [np.quantile(fft.numpy(), 0.05, axis=(0,1,2,3,4,5))]
    lineQuantileUpper += [np.quantile(fft.numpy(), 0.95, axis=(0,1,2,3,4,5))]



fig, ax = plt.subplots(1, figsize=(4.5,1.8), dpi=150)
ax.text(1-0.008, 1-0.018, getDatasetName(datasetName), color="k", bbox=dict(facecolor="whitesmoke", edgecolor="darkslategray", boxstyle="round"),
        horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)

if datasetName in ["extrap", "interp", "longer", "highRey", "lowRey", "varReyIn"]:
    ax.set_xlabel("Wavenumber $\kappa$ along vertical line downstream")
elif datasetName in ["zInterp"]:
    ax.set_xlabel("Wavenumber of %s for Flow along Vertical Line at x Position %1.2f" % (field, cylsDownstream * prediction.shape[5] * 2 * ((2*3.1415) / 1024) ))

ax.set_ylabel("$\\overline{v_x}$ Amplitude $*\kappa^4$")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
#ax.yaxis.grid(True, which="both")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
for i in range(len(modelNames)):
    lineMean[i] = lineMean[i] * freq * freq * freq * freq
    lineStd[i] = lineStd[i] * freq * freq * freq * freq
    lineQuantileLower[i] = lineQuantileLower[i] * freq * freq * freq * freq
    lineQuantileUpper[i] = lineQuantileUpper[i] * freq * freq * freq * freq
    color = getColor(modelNames[i])
    label = getModelName(modelNames[i])
    if modelNames[i] == "Simulation":
        ms = np.logspace(5, 3, freq.shape[0], base=2)
        ax.plot(freq, lineMean[i], linewidth=1.5, color=color, linestyle="dotted")
        ax.scatter(freq, lineMean[i], ms, color=color, marker="o")
        ax.plot([], [], linewidth=1.5, color=color, label=label, linestyle="dotted", marker="o", markersize=4)
    else:
        ax.plot(freq, lineMean[i], linewidth=1.5, color=color, label=label)
        if useStd:
            #ax.fill_between(freq, lineMean[i] - lineStd[i], lineMean[i] + lineStd[i], facecolor=color, alpha=0.15)
            ax.fill_between(freq, lineQuantileLower[i], lineQuantileUpper[i], facecolor=color, alpha=0.15)
#ax.legend()

fig.tight_layout(pad=0.4)
fig.savefig("%s/downstream_line_freq_%s_%s.pdf" % (outputFolder, datasetName, field))




