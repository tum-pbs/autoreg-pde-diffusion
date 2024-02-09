import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from lsim.distance_model import DistanceModel as LSIM_Model
from turbpred.loss import loss_lsim

from plot_color_and_name_mapping import getColor, getModelName, getDatasetName, getFieldIndex, getLossRelevantFields, getColormapAndNorm

plt.rcParams['pdf.fonttype'] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams['ps.fonttype'] = 42

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




datasetName = "varReyIn"
setupName = "all"
modelMinMax = (0,3)
evalMinMax = (0,5)
sequenceMinMax = (0,16)
timeMinMax = (0,250)

predictionFolder = "results/sampling/%s" % datasetName
outputFolder = "results"

models = {
    #"ResNet": "resnet-m2.npz",
    "Dil-ResNet": "dil-resnet-m2.npz",

    "FNO16": "fno-16modes-m2.npz",
    #"FNO32": "fno-32modes-m2.npz",

    #"TF-MGN": "tf-mgn.npz",
    "TF-Enc": "tf-enc.npz",
    #"TF-VAE": "tf-vae.npz",

    "U-Net": "unet-m2.npz",
    "U-Net-ut": "unet-m8.npz",
    "U-Net-tn": "unet-m2-noise0.01.npz",

    "Refiner": "refiner-r4_std%s.npz" % ("0.00001" if datasetName in ["zInterp"] else "0.000001"),

    "ACDM-ncn": "acdm20_ncn.npz",
    "ACDM": "acdm20.npz",
}

metric = "PCC"
withInset = False


groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
if "obsMask" in groundTruthDict:
    obsMask = groundTruthDict["obsMask"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
    groundTruth = groundTruth * obsMask # ignore obstacle area
print("Original ground truth shape: %s" % (str(list(groundTruth.shape))))
groundTruth = groundTruth[:,:,
                        sequenceMinMax[0]:sequenceMinMax[1],
                        timeMinMax[0]:timeMinMax[1],
                        getLossRelevantFields(datasetName)[0]:getLossRelevantFields(datasetName)[1]]
print("Loaded ground truth with shape: %s" % (str(list(groundTruth.shape))))

if metric == "LSIM":
    lsimModel = LSIM_Model(baseType="lsim", isTrain=False, useGPU=True)
    lsimModel.load("src/lsim/models/LSiM.pth")

modelNames = []
distanceMean = []
distanceStd = []

for modelName, modelPath in models.items():
    modelNames += [modelName]

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
        mse = F.mse_loss(groundTruth.expand_as(prediction), prediction, reduction="none")
        mseOverTime = torch.mean(mse, dim=(4,5,6))
        distanceMean += [torch.mean(mseOverTime, dim=(0,1,2)).numpy()]
        distanceStd += [torch.std(mseOverTime, dim=(0,1,2)).numpy()]

    elif metric == "PCC":
        gtReshape = groundTruth.expand_as(prediction)
        gtReshape = torch.reshape(gtReshape, (gtReshape.shape[0],gtReshape.shape[1],gtReshape.shape[2],gtReshape.shape[3],-1))
        gtReshape = torch.reshape(gtReshape, (-1,gtReshape.shape[4]))

        predReshape = torch.reshape(prediction, (prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3],-1))
        predReshape = torch.reshape(predReshape, (-1,predReshape.shape[4]))
        cor = []
        for i in range(gtReshape.shape[0]):
            stacked = torch.concat([gtReshape[i:i+1], predReshape[i:i+1]], dim=0)
            result = torch.corrcoef(stacked)[0,1]
            cor += [result.item()]
        cor = torch.tensor(cor)
        cor = torch.reshape(cor, (prediction.shape[0],prediction.shape[1],prediction.shape[2],prediction.shape[3]))

        distanceMean += [torch.mean(cor, dim=(0,1,2)).numpy()]
        distanceStd += [torch.std(cor, dim=(0,1,2)).numpy()]

    elif metric == "LSIM":
        pred = torch.reshape(prediction, [-1, prediction.shape[3], prediction.shape[4], prediction.shape[5], prediction.shape[6]])
        gt = groundTruth.expand_as(prediction)
        gt = torch.reshape(gt, [-1, gt.shape[3], gt.shape[4], gt.shape[5], gt.shape[6]])
        lsim = []
        for i in range(pred.shape[0]):
            with torch.no_grad():
                lsim += [loss_lsim(lsimModel, gt[i:i+1].cuda(), pred[i:i+1].cuda()).cpu()]
        lsim = torch.concat(lsim, dim=0)
        lsim = torch.reshape(lsim, [prediction.shape[0], prediction.shape[1], prediction.shape[2], lsim.shape[1], lsim.shape[2]])
        lsimOverTime = torch.mean(lsim, dim=(4))
        distanceMean += [torch.mean(lsimOverTime, dim=(0,1,2)).numpy()]
        distanceStd += [torch.std(lsimOverTime, dim=(0,1,2)).numpy()]


fig, ax = plt.subplots(1, figsize=(5.0,2.3), dpi=150)
#ax.set_title(title)
ax.text(0.008, 0.018, getDatasetName(datasetName), color="k", bbox=dict(facecolor="whitesmoke", edgecolor="darkslategray", boxstyle="round"),
        horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
ax.set_xlabel("Time step")
#ax.set_ylabel(metric)
ax.set_ylabel("Correlation to Sim.")
ax.yaxis.grid(True)
ax.set_axisbelow(True)

if withInset:
    axIns = ax.inset_axes([0.05, 0.2, 0.45, 0.50], xlim=(0.0, 100), ylim=(0.99, 1.00))
    mark_inset(ax, axIns, loc1=1, loc2=3, fc="none", ec="0.5")
    axIns.tick_params(axis="y", labelsize=8)
    axIns.set_facecolor("0.95")
    axIns.set_xticks([])


ax.set_ylim([0.94,1.002])
for i in range(len(modelNames)):
    linestyle = "dashed" if modelNames[i] == "Simulation" else "solid"
    color = getColor(modelNames[i])
    label = getModelName(modelNames[i])
    ax.plot(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i], linewidth=1.5, color=color, label=label, linestyle=linestyle)
    ax.fill_between(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i] - distanceStd[i], distanceMean[i] + distanceStd[i], facecolor=color, alpha=0.15)
    if withInset:
        axIns.plot(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i], linewidth=1.5, color=color, label=label, linestyle=linestyle)
        axIns.fill_between(np.arange(distanceMean[i].shape[0]) + 2, distanceMean[i] - distanceStd[i], distanceMean[i] + distanceStd[i], facecolor=color, alpha=0.15)

#ax.legend(ncol=3, columnspacing=0.8, loc="lower left", fontsize=9.5)#bbox_to_anchor=(1.01, 1.06))

fig.tight_layout(pad=0.4)
fig.savefig("%s/loss_sequence_%s_%s.pdf" % (outputFolder, datasetName, metric.lower()))




