import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42




datasetName = "longer"
modelMinMax = (2,3)
evalMinMax = (0,5)
sequence = [0]
timeSteps = [79,129]
spatialZoom = [[20,70], [7,57]]
field = "pres"

predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"

models = {
    "Simulation": "groundTruth.dict",
    #"TF-VAE": "tf-vae.npz",
    #"Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),
    "ACDM": "acdm20.npz",

}


modelNames = []
frameData = []

for modelName, modelPath in models.items():
    modelNames += [modelName]

    if modelPath == "groundTruth.dict":
        groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
        groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
        obsMask = groundTruthDict["obsMask"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
        groundTruth = groundTruth * obsMask # ignore obstacle area
        print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
        prediction = groundTruth[:,:,:,:,:, spatialZoom[0][0]:spatialZoom[0][1], spatialZoom[1][0]:spatialZoom[1][1]]
        prediction = prediction[:,:,:,:, getFieldIndex(datasetName, field)]
        prediction = prediction[:,:,:, timeSteps]
        prediction = prediction[:,:, sequence]
        prediction = torch.squeeze(prediction, dim=2).squeeze(dim=0)
        print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

    else:
        fullPath = os.path.join(predictionFolder, modelPath)
        prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
        prediction = prediction * obsMask
        prediction = prediction[modelMinMax[0]:modelMinMax[1],
                                evalMinMax[0]:evalMinMax[1],
                                :,:,:,
                                spatialZoom[0][0]:spatialZoom[0][1],
                                spatialZoom[1][0]:spatialZoom[1][1]]
        prediction = prediction[:,:,:,:, getFieldIndex(datasetName, field)]
        prediction = prediction[:,:,:, timeSteps]
        prediction = prediction[:,:, sequence]
        prediction = torch.squeeze(prediction, dim=2).squeeze(dim=0)
        print("Loaded prediction from model %s with shape: %s" % (modelName, str(list(prediction.shape))))

    if field == "vort":
        vxDx, vxDy = torch.gradient(prediction[:,:,0], dim=[2,3])
        vyDx, vyDy = torch.gradient(prediction[:,:,1], dim=[2,3])
        prediction = vyDx - vxDy # curl == vorticity

    frameData += [prediction.permute(0,1,3,2).numpy()]



fig, axs = plt.subplots(nrows=len(timeSteps), ncols=5, figsize=(5.8,2.4), dpi=250)
for i in range(len(timeSteps)):
    for j in range(5):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

axs[len(timeSteps)-1,0].set_xlabel(r"Sim. ($%d\times%d$)" % (spatialZoom[0][1]-spatialZoom[0][0], spatialZoom[1][1]-spatialZoom[1][0]))
for i in range(1,4):
    axs[len(timeSteps)-1,i].set_xlabel("Sample %d" % i)
axs[len(timeSteps)-1,4].set_xlabel("Std. Dev.")

for t in range(len(timeSteps)):
    axs[t,0].set_ylabel("$t=%d$" % (timeSteps[t]+1))
    cmap, norm = getColormapAndNorm(datasetName, field)
    im = axs[t,0].imshow(frameData[0][0,t], interpolation="catrom", cmap=cmap, norm=norm)
    for i in range(3):
        im = axs[t,i+1].imshow(frameData[1][i,t], interpolation="catrom", cmap=cmap, norm=norm)
    im = axs[t,4].imshow(np.std(frameData[1][:,t], axis=0), interpolation="catrom", cmap="YlGnBu", vmin=0, vmax=0.16)


fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.2)
fig.subplots_adjust(right=0.88, top=0.92)
fig.suptitle(getModelName(modelNames[1]))
cbarAx = fig.add_axes([0.90, 0.04, 0.025, 0.92])
cbarAx.tick_params(labelsize=8)
fig.colorbar(im, cax=cbarAx)
fig.savefig("%s/data_posterior_%s_%s.pdf" % (outputFolder, datasetName, field))




