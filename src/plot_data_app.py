import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42




datasetName = "zInterp"
modelMinMax = (0,1)
evalMinMax = (0,1)
#sequenceMinMax = (0,1)
sequenceMinMax = (8,9)
timeSteps = [4,29,84,139,194,249] if datasetName in ["lowRey", "highRey", "varReyIn"] else \
            [4,29,79,129,179,229] if datasetName in ["interp", "extrap", "longer"] else \
            [4,19,39,59,79,99]
field = "vort"


predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"

models = {
    "Simulation": "groundTruth.dict",
    "ResNet": "resnet-m2.npz",
    "Dil-ResNet": "dil-resnet-m2.npz",

    "FNO16": "fno-16modes-m2.npz",
    "FNO32": "fno-32modes-m2.npz",

    "TF-MGN": "tf-mgn.npz",
    "TF-Enc": "tf-enc.npz",
    "TF-VAE": "tf-vae.npz",

    "U-Net": "unet-m2.npz",
    "U-Net-ut": "unet-m8.npz",
    "U-Net-tn": "unet-m2-noise0.01.npz",

    "Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),

    "ACDM-ncn": "acdm%d_ncn.npz" % (100 if datasetName in ["zInterp"] else 20),
    "ACDM-bold": "acdm%d.npz" % (100 if datasetName in ["zInterp"] else 20),
}


modelNames = []
frameData = []

for modelName, modelPath in models.items():
    modelNames += [modelName]

    if modelPath == "groundTruth.dict":
        groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
        groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
        #obsMask = groundTruthDict["obsMask"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
        #groundTruth = groundTruth * obsMask # ignore obstacle area
        print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
        prediction = groundTruth[:,:,
                                sequenceMinMax[0]:sequenceMinMax[1],
                                timeSteps]
        prediction = torch.squeeze(prediction[:,:,:,:,getFieldIndex(datasetName, field)])
        print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

    else:
        fullPath = os.path.join(predictionFolder, modelPath)
        prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
        #prediction = prediction * obsMask
        prediction = prediction[modelMinMax[0]:modelMinMax[1],
                            evalMinMax[0]:evalMinMax[1],
                            sequenceMinMax[0]:sequenceMinMax[1],
                            timeSteps]
        prediction = torch.squeeze(prediction[:,:,:,:,getFieldIndex(datasetName, field)])
        print("Loaded prediction from model %s with shape: %s" % (modelName, str(list(prediction.shape))))

    if field == "vort":
        vxDx, vxDy = torch.gradient(prediction[:,0], dim=[1,2])
        vyDx, vyDy = torch.gradient(prediction[:,1], dim=[1,2])
        prediction = vyDx - vxDy # curl == vorticity

    frameData += [prediction.permute(0,2,1).numpy()]



fig, axs = plt.subplots(nrows=len(modelNames), ncols=len(timeSteps), figsize=(9.5,13.5), dpi=250, squeeze=False)
for i in range(len(modelNames)):
    for j in range(len(timeSteps)):
        if i == len(modelNames)-1:
            axs[i,j].set_xlabel("$t=%s$" % (timeSteps[j]+1))
        if j == 0:
            axs[i,j].set_ylabel(getModelName(modelNames[i]))
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
        cmap, norm = getColormapAndNorm(datasetName, field)
        im = axs[i,j].imshow(frameData[i][j], interpolation="nearest", cmap=cmap, norm=norm)

fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.3)
#fig.subplots_adjust(bottom=0.9)
#cbarAx = fig.add_axes([0.04, 0.92, 0.92, 0.025])
#fig.colorbar(im, cax=cbarAx, orientation="horizontal")
#cbarAx.tick_params(labelsize=8)
fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.03, aspect=50, shrink=0.98, pad=0.03)
fig.savefig("%s/data_%s_%s.pdf" % (outputFolder, datasetName, field))




