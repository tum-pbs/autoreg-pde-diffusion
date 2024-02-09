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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"




outputFolder = "results"

datasets = ["lowRey", "highRey", "extrap", "interp", "zInterp"]
metric = "MSE"

legend = True
load = False
save = True

if metric == "MSE":
    rescaleMap = {"lowRey": 1e4, "highRey": 1e5, "extrap": 1e3, "interp": 1e3, "zInterp": 1e2}
    yLabelMap = {"lowRey": "$(10^{-4})$", "highRey": "$(10^{-5})$", "extrap": "$(10^{-3})$", "interp": "$(10^{-3})$", "zInterp": "$(10^{-2})$"}
    yLimitMap = {"lowRey": [0,11], "highRey": [0,10.9], "extrap": [0,7.6], "interp": [0,11], "zInterp": [0,20]}

else:
    rescaleMap = {"lowRey": 1e2, "highRey": 1e2, "extrap": 1e1, "interp": 1e1, "zInterp": 1e1}
    yLabelMap = {"lowRey": "$(10^{-2})$", "highRey": "$(10^{-2})$", "extrap": "$(10^{-1})$", "interp": "$(10^{-1})$", "zInterp": "$(10^{-1})$"}
    yLimitMap = {"lowRey": [0,25.5], "highRey": [0,7.6], "extrap": [0,5.0], "interp": [0,5.0], "zInterp": [0,12]}


modelNames = {}
distanceMean = {}
distanceStd = {}

if not load:
    if metric == "LSIM":
        lsimModel = LSIM_Model(baseType="lsim", isTrain=False, useGPU=True)
        lsimModel.load("src/lsim/models/LSiM.pth")

    for datasetName in datasets:

        modelMinMax = (0,3)
        evalMinMax = (0,5)
        sequenceMinMax = (0,16)
        timeMinMax = (0,240)

        predictionFolder = "results/sampling/%s" % datasetName

        models = {
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
            "ACDM": "acdm%d.npz" % (100 if datasetName in ["zInterp"] else 20),
        }


        groundTruthDict = torch.load(os.path.join(predictionFolder, "groundTruth.dict"))
        groundTruth = groundTruthDict["data"].unsqueeze(0).unsqueeze(0)
        if "obsMask" in groundTruthDict:
            obsMask = groundTruthDict["obsMask"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
            groundTruth = groundTruth * obsMask # ignore obstacle area
        print("%s:   Original ground truth shape: %s" % (datasetName, str(list(groundTruth.shape))))
        groundTruth = groundTruth[:,:,
                                sequenceMinMax[0]:sequenceMinMax[1],
                                timeMinMax[0]:timeMinMax[1],
                                getLossRelevantFields(datasetName)[0]:getLossRelevantFields(datasetName)[1]]
        print("%s:   Loaded ground truth with shape: %s" % (datasetName, str(list(groundTruth.shape))))

        modelNames[datasetName] = []
        distanceMean[datasetName] = []
        distanceStd[datasetName] = []

        for modelName, modelPath in models.items():
            modelNames[datasetName] += [modelName]

            fullPath = os.path.join(predictionFolder, modelPath)
            if not os.path.exists(fullPath):
                distanceMean[datasetName] += [0]
                distanceStd[datasetName] += [0]
                print("Skipping %s" % fullPath)
                continue
            prediction = torch.from_numpy(np.load(fullPath)["arr_0"])
            if "obsMask" in groundTruthDict:
                prediction = prediction * obsMask
            prediction = prediction[modelMinMax[0]:modelMinMax[1],
                                evalMinMax[0]:evalMinMax[1],
                                sequenceMinMax[0]:sequenceMinMax[1],
                                timeMinMax[0]:timeMinMax[1],
                                getLossRelevantFields(datasetName)[0]:getLossRelevantFields(datasetName)[1]]
            print("%s:   Loaded prediction from model %s with shape: %s" % (datasetName, modelName, str(list(prediction.shape))))

            if metric == "MSE":
                mse = F.mse_loss(groundTruth.expand_as(prediction), prediction, reduction="none")
                mseScalar = torch.mean(mse, dim=(3,4,5,6))
                distanceMean[datasetName] += [rescaleMap[datasetName] * torch.mean(mseScalar).numpy()]
                distanceStd[datasetName] += [rescaleMap[datasetName] * torch.std(mseScalar).numpy()]

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
                lsimScalar = torch.mean(lsim, dim=(3,4))
                distanceMean[datasetName] += [rescaleMap[datasetName] * torch.mean(lsimScalar).numpy()]
                distanceStd[datasetName] += [rescaleMap[datasetName] * torch.std(lsimScalar).numpy()]

    if save:
        torch.save(modelNames, "results/temp/%s_ModelNames.loss" % metric)
        torch.save(distanceMean, "results/temp/%s_DistanceMean.loss" % metric)
        torch.save(distanceStd, "results/temp/%s_DistanceStd.loss" % metric)

else:
    modelNames = torch.load("results/temp/%s_ModelNames.loss" % metric)
    distanceMean = torch.load("results/temp/%s_DistanceMean.loss" % metric)
    distanceStd= torch.load("results/temp/%s_DistanceStd.loss" % metric)


fig, axs = plt.subplots(ncols=len(datasets), figsize=(16,1.2), dpi=150)

for i in range(len(datasets)):
    datasetName = datasets[i]
    if not legend:
        axs[i].set_title(getDatasetName(datasetName))
    axs[i].set_ylabel("%s %s" % (metric, yLabelMap[datasetName]))
    axs[i].set_xticks([])
    axs[i].yaxis.grid(True)
    #axs[i].ticklabel_format(style='sci', axis="y", scilimits=(0,0))
    axs[i].set_axisbelow(True)
    #axs[i].set_yscale("log", base=10)
    axs[i].set_ylim(yLimitMap[datasetName])
    colors = [getColor(k) for k in modelNames[datasetName]]

    posX = [0.0,1.0, 2.5,3.5, 5.0,6.0,7.0, 8.5,9.5,10.5, 12.0, 13.5,14.5]
    axs[i].set_xlim([-0.8,15.3])
    legHandle = axs[i].bar(posX, distanceMean[datasetName], 1.0, yerr=distanceStd[datasetName], capsize=2, color=colors)

for i in range(len(modelNames[datasetName])):
    print()
    for j in range(len(datasets)):
        print("%s - %s - %s: %1.1f +- %1.1f" % ("{:<4}".format(metric), "{:<7}".format(datasets[j]), "{:<10}".format(modelNames[datasets[j]][i]), distanceMean[datasets[j]][i], distanceStd[datasets[j]][i]))

if legend:
    labels = [getModelName(k) for k in modelNames[datasetName]]
    fig.legend(legHandle, labels, ncol=len(labels), columnspacing=1.0, loc="upper center", bbox_to_anchor=(0.5, 0.07))

#fig.tight_layout(pad=0.4)
fig.subplots_adjust(wspace=0.3)
fig.savefig("%s/loss_all_%s.pdf" % (outputFolder, metric.lower()), bbox_inches="tight")

print("\nPlot complete.")




