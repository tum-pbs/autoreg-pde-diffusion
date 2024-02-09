import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42




datasetName = "varReyIn"
modelMinMax = (1,2)
evalMinMax = (0,1)
sequenceMinMax = (0,1)
#sequenceMinMax = (6,7)
timeMinMax = (0,250)
spatialZoom = [[20,84], [0,64]] if datasetName in ["lowRey", "highRey", "varReyIn"] else \
              [[6,70], [0,64]] if datasetName in ["interp", "extrap", "longer"] else \
              [[0,64], [0,64]]
#spatialZoom = [[0,128], [0,64]] if datasetName in ["lowRey", "highRey", "varReyIn"] else \
#              [[0,128], [0,64]] if datasetName in ["interp", "extrap", "longer"] else \
#              [[0,128], [0,64]]
field = "vort"

predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"


models = {
    "Simulation": "groundTruth.dict",

    #"ResNet": "resnet-s2.npz",
    "Dil-ResNet": "dil-resnet-s2.npz",

    "FNO16": "fno-16modes-s2.npz",
    "FNO32": "fno-32modes-s2.npz",

    #"TF-MGN": "tf-mgn.npz",
    "TF-Enc": "tf-enc.npz",
    "TF-VAE": "tf-vae.npz",

    "U-Net": "unet-s2.npz",
    "U-Net-ut": "unet-s8.npz",
    "U-Net-tn": "unet-s2-noise0.01.npz",

    "Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),

    "ACDM-ncn": "acdm%d_ncn.npz" % (100 if datasetName in ["zInterp"] else 20),
    "ACDM": "acdm%d.npz" % (100 if datasetName in ["zInterp"] else 20),
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
                                timeMinMax[0]:timeMinMax[1],
                                :,
                                spatialZoom[0][0]:spatialZoom[0][1],
                                spatialZoom[1][0]:spatialZoom[1][1]]
        prediction = torch.squeeze(prediction[:,:,:,:,getFieldIndex(datasetName, field)])
        print("Loaded ground truth with shape: %s" % (str(list(prediction.shape))))

    else:
        fullPath = os.path.join(predictionFolder, modelPath)
        prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
        #prediction = prediction * obsMask
        prediction = prediction[modelMinMax[0]:modelMinMax[1],
                            evalMinMax[0]:evalMinMax[1],
                            sequenceMinMax[0]:sequenceMinMax[1],
                            timeMinMax[0]:timeMinMax[1],
                            :,
                            spatialZoom[0][0]:spatialZoom[0][1],
                            spatialZoom[1][0]:spatialZoom[1][1]]
        prediction = torch.squeeze(prediction[:,:,:,:,getFieldIndex(datasetName, field)])
        print("Loaded prediction from model %s with shape: %s" % (modelName, str(list(prediction.shape))))

    if field == "vort":
        vxDx, vxDy = torch.gradient(prediction[:,0], dim=[1,2])
        vyDx, vyDy = torch.gradient(prediction[:,1], dim=[1,2])
        prediction = vyDx - vxDy # curl == vorticity

    frameData += [prediction.permute(0,2,1).numpy()]

frameData = np.array(frameData)


frames = []
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(8,4.5), dpi=100)
axs = axs.reshape(-1)
fig.patch.set_facecolor("lightgray")

for t in range(frameData.shape[1]):
    ims = []
    for i in range(len(modelNames)):
        axs[i].set_ylabel(getModelName(modelNames[i]))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        cmap, norm = getColormapAndNorm(datasetName, field)
        im = axs[i].imshow(frameData[i][t], interpolation="catrom", cmap=cmap, norm=norm)
        ims += [im]
    frames += [ims]

fig.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.2)
fig.subplots_adjust(right=0.88)
cbarAx = fig.add_axes([0.90, 0.06, 0.025, 0.88])
fig.colorbar(im, cax=cbarAx)
cbarAx.tick_params(labelsize=10)

anim = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
anim.save("%s/data_%s_%s.mp4" % (outputFolder, datasetName, field))



