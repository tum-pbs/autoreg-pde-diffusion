import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lsim.distance_model import DistanceModel as LSIM_Model
from turbpred.loss import loss_lsim

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




datasetName = "zInterp"
modelMinMax = (0,3)
evalMinMax = (0,5)
sequenceMinMax = (0,16)
timeMinMax = (0,240)

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

    "ACDM-ncn": "acdm%d_ncn.npz" % (100 if datasetName in ["zInterp"] else 20),
    "ACDM": "acdm%d.npz" % (100 if datasetName in ["zInterp"] else 20),
}

metric = "L1"
useStd = False


modelNames = []
distanceMean = []
distanceStd = []

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

    if metric == "MSE":
        error = F.mse_loss(prediction[:, :, :, 0:prediction.shape[3]-1], prediction[:, :, :, 1:prediction.shape[3]], reduction="none")
    if metric == "L1":
        error = F.l1_loss(prediction[:, :, :, 0:prediction.shape[3]-1], prediction[:, :, :, 1:prediction.shape[3]], reduction="none")

    errorOverTime = torch.mean(error, dim=(4,5,6))
    errorOverTime = errorOverTime[:,:,:,1:]
    distanceMean += [torch.mean(errorOverTime, dim=(0,1,2)).numpy()]
    distanceStd += [torch.std(errorOverTime, dim=(0,1,2)).numpy()]


fig, ax = plt.subplots(1, figsize=(5.0,2.5), dpi=150)
#ax.set_title(getDatasetName(datasetName))
ax.text(0.008, 0.018, getDatasetName(datasetName), color="k", bbox=dict(facecolor="whitesmoke", edgecolor="darkslategray", boxstyle="round"),
        horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)

ax.set_xlabel("Time step $t$")
#ax.set_ylabel(metric)
ax.set_ylabel(r"$\Vert \, (s^{t} - s^{t-1}) / \Delta t \, \Vert_2^2$" if metric == "MSE" else r"$\Vert \, (s^{t} - s^{t-1}) / \Delta t \, \Vert_1$")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
#ax.set_ylim([0.005,0.025])
#ax.set_ylim([0.003,0.021])
#ax.set_ylim([0.012,0.021])
#ax.set_ylim([0.0,0.05])
#ax.set_ylim([0.0,0.025])
ax.set_ylim([-0.005,0.05])
#ax.set_ylim([0.0,0.03])
for i in range(len(modelNames)):
    color = getColor(modelNames[i])
    label = getModelName(modelNames[i])
    if modelNames[i] == "Simulation":
        ax.plot(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i], linewidth=1.5, color=color, label=label, linestyle="dashed")
        if useStd:
            ax.fill_between(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i] - distanceStd[i], distanceMean[i] + distanceStd[i], facecolor=color, alpha=0.15)
    else:
        ls = "dashdot" if "Pre" in modelNames[i] else "solid"
        ax.plot(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i], linewidth=1.5, color=color, label=label, linestyle=ls)
        if useStd:
            ax.fill_between(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i] - distanceStd[i], distanceMean[i] + distanceStd[i], facecolor=color, alpha=0.15)
#ax.legend(ncol=4, columnspacing=0.8, loc="upper center", fontsize=6)

fig.tight_layout(pad=0.4)
fig.savefig("%s/loss_to_prev_%s_%s.pdf" % (outputFolder, datasetName, metric.lower()))




