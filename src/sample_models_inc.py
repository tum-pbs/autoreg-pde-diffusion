import os, time, sys
import shutil
import copy
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np

from turbpred.model import PredictionModel
from turbpred.model_diffusion import DiffusionModel
from turbpred.params import DataParams
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms


device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


outFolder = "results/sampling"

testSets = {
    "lowRey":
        TurbulenceDataset("Test Low Reynolds 100-200", ["data"], filterTop=["128_inc"], filterSim=[[82,84,86,88,90]],
                filterFrame=[(1000,1150)], sequenceLength=[[60,2]], simFields=["pres"], simParams=["rey"], printLevel="sim"),
    "highRey" :
        TurbulenceDataset("Test High Reynolds 900-1000", ["data"], filterTop=["128_inc"], filterSim=[[0,2,4,6,8]],
                filterFrame=[(1000,1150)], sequenceLength=[[60,2]], simFields=["pres"], simParams=["rey"], printLevel="sim"),
    "varReyIn" :
        TurbulenceDataset("Test Varying Reynolds Number (200-900)", ["data"], filterTop=["128_reyVar"], filterSim=[[0]],
                filterFrame=[(300,800)], sequenceLength=[[250,2]], simFields=["pres"], simParams=["rey"], printLevel="sim"),
}
dataTransformationFieldIndices = [0,1,3,4] # ORDER (fields): velocity (x,y), velocity z / density, pressure, ORDER (params): rey, mach, zslice

numEvals = 5

models = {
    "tf-enc":
        (
            ["runs/2D_Inc/128_tf-enc_MSE_00/Model.pth",
            "runs/2D_Inc/128_tf-enc_MSE_01/Model.pth",
            "runs/2D_Inc/128_tf-enc_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),

    "tf-vae":
        (
            ["runs/2D_Inc/128_tf-vae_MSE_00/Model.pth",
            "runs/2D_Inc/128_tf-vae_MSE_01/Model.pth",
            "runs/2D_Inc/128_tf-vae_MSE_MSE_02/Model.pth",
            ],
            {"lowRey": 4, "highRey": 4, "varReyIn": 1},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"lowRey": False, "highRey": False, "varReyIn": True}
            },
        ),

    "tf-mgn":
        (
            ["runs/2D_Inc/128_tf-mgn_MSE_00/Model.pth",
            "runs/2D_Inc/128_tf-mgn_MSE_01/Model.pth",
            "runs/2D_Inc/128_tf-mgn_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),


    "unet-m2":
        (
            ["runs/2D_Inc/128_unet-m2_MSE_00/Model.pth",
            "runs/2D_Inc/128_unet-m2_MSE_01/Model.pth",
            "runs/2D_Inc/128_unet-m2_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),

    "unet-m8":
        (
            ["runs/2D_Inc/128_unet-m8_MSE_00/Model.pth",
            "runs/2D_Inc/128_unet-m8_MSE_01/Model.pth",
            "runs/2D_Inc/128_unet-m8_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),

    "unet-m2-noise0.01":
        (
            ["runs/2D_Inc/128_unet-m2-noise0.01_MSE_00/Model.pth",
            "runs/2D_Inc/128_unet-m2-noise0.01_MSE_01/Model.pth",
            "runs/2D_Inc/128_unet-m2-noise0.01_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),


    "refiner4_std0.000001":
       (
            ["runs/2D_Inc/128_refiner4_std0.000001_00/Model.pth",
            "runs/2D_Inc/128_refiner4_std0.000001_01/Model.pth",
            "runs/2D_Inc/128_refiner4_std0.000001_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"lowRey": True, "highRey": True, "varReyIn": True},
           },
       ),


    "resnet-m2":
        (
            ["runs/2D_Inc/128_resnet-m2_MSE_00/Model.pth",
            "runs/2D_Inc/128_resnet-m2_MSE_01/Model.pth",
            "runs/2D_Inc/128_resnet-m2_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),

    "dil-resnet-m2":
        (
            ["runs/2D_Inc/128_dil-resnet-m2_MSE_00/Model.pth",
            "runs/2D_Inc/128_dil-resnet-m2_MSE_01/Model.pth",
            "runs/2D_Inc/128_dil-resnet-m2_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),


    "fno-16modes-m2":
        (
            ["runs/2D_Inc/128_fno-16modes-m2_MSE_00/Model.pth",
            "runs/2D_Inc/128_fno-16modes-m2_MSE_01/Model.pth",
            "runs/2D_Inc/128_fno-16modes-m2_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),

    "fno-32modes-m2":
        (
            ["runs/2D_Inc/128_fno-32modes-m2_MSE_00/Model.pth",
            "runs/2D_Inc/128_fno-32modes-m2_MSE_01/Model.pth",
            "runs/2D_Inc/128_fno-32modes-m2_MSE_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 1},
            {"numEvals": 1,
            },
        ),


    "direct-ddpm+Prev20":
       (
            ["runs/2D_Inc/128_direct-ddpm+Prev20_00/Model.pth",
            "runs/2D_Inc/128_direct-ddpm+Prev20_01/Model.pth",
            "runs/2D_Inc/128_direct-ddpm+Prev20_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"lowRey": True, "highRey": True, "varReyIn": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "noisy"
           },
       ),

    "direct-ddpm+Prev20_ncn":
       (
            ["runs/2D_Inc/128_direct-ddpm+Prev20_ncn_00/Model.pth",
            "runs/2D_Inc/128_direct-ddpm+Prev20_ncn_01/Model.pth",
            "runs/2D_Inc/128_direct-ddpm+Prev20_ncn_02/Model.pth",
            ],
            {"lowRey": 8, "highRey": 8, "varReyIn": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"lowRey": True, "highRey": True, "varReyIn": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "clean"
           },
       ),

}


os.makedirs(outFolder, exist_ok=True)

# SAVE GROUND TRUTH
for shortTestSetName, testSet in testSets.items():
    testSetOutPath = os.path.join(outFolder, shortTestSetName)
    os.makedirs(testSetOutPath, exist_ok=True)

    shutil.copy(sys.argv[0], os.path.join(testSetOutPath, "sample_models.py")) # copy this script to output folder for reference

    if os.path.isfile(os.path.join(testSetOutPath, "groundTruth.dict")):
        print("Skipping %s" % os.path.join(testSetOutPath, "groundTruth.dict"))
        continue

    testSet.transform = lambda x: x
    testSampler = SequentialSampler(testSet)
    testLoader = DataLoader(testSet, sampler=testSampler, batch_size=len(testSet), drop_last=False)

    sample = next(iter(testLoader))

    torch.save(sample, os.path.join(testSetOutPath, "groundTruth.dict"))
    print("Saved ground truth to %s" % os.path.join(testSetOutPath, "groundTruth.dict"))
    # output shape:
    # [data set sequences,  sequence length,  channels,  positionX,  positionY]


# SAVE PREDICTIONS
for modelName, modelData in models.items():
    print("\n\n--------------------------------------------------")
    print("Processing model: %s" % (modelName))
    print("--------------------------------------------------")

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)

    modelPaths = modelData[0]
    batchDict = modelData[1]
    evalOptions = modelData[2]

    for shortTestSetName, testSet in testSets.items():
        print("\nDATASET: %s" % (shortTestSetName))
        testSetOutPath = os.path.join(outFolder, shortTestSetName)
        os.makedirs(testSetOutPath, exist_ok=True)
        if os.path.isfile(os.path.join(testSetOutPath, modelName + ".npz")):
            print("Skipping %s" % os.path.join(testSetOutPath, modelName + ".npz"))
            continue

        predFull = []

        timerFullStart = time.perf_counter()
        for modelPath in modelPaths:
            model = PredictionModel.load(modelPath, useGPU=device=="cuda")
            model.eval()

            if isinstance(model.modelDecoder, DiffusionModel):
                if "samplingMode" in evalOptions:
                    model.modelDecoder.inferenceSamplingMode = evalOptions["samplingMode"]
                if "posteriorSampling" in evalOptions:
                    model.modelDecoder.inferencePosteriorSampling = evalOptions["posteriorSampling"]
                if "initialSampling" in evalOptions:
                    model.modelDecoder.inferenceInitialSampling = evalOptions["initialSampling"]
                if "conditioningIntegration" in evalOptions:
                    model.modelDecoder.inferenceConditioningIntegration = evalOptions["conditioningIntegration"]

            elif isinstance(model.modelDecoder, torch.nn.ModuleList):
                for module in model.modelDecoder:
                    if isinstance(module, DiffusionModel):
                        if "samplingMode" in evalOptions:
                            module.inferenceSamplingMode = evalOptions["samplingMode"]
                        if "posteriorSampling" in evalOptions:
                            module.inferencePosteriorSampling = evalOptions["posteriorSampling"]
                        if "initialSampling" in evalOptions:
                            module.inferenceInitialSampling = evalOptions["initialSampling"]
                        if "conditioningIntegration" in evalOptions:
                            module.inferenceConditioningIntegration = evalOptions["conditioningIntegration"]

            batchSize = batchDict[shortTestSetName]

            p_d_test = copy.deepcopy(model.p_d)
            p_d_test.augmentations = ["normalize"]
            p_d_test.sequenceLength = testSet.sequenceLength
            p_d_test.randSeqOffset = False
            testTransformations = Transforms(p_d_test)
            testSet.transform = testTransformations

            testSampler = SequentialSampler(testSet)
            testLoader = DataLoader(testSet, sampler=testSampler, batch_size=batchSize, drop_last=False)#, num_workers=4)

            with torch.no_grad():
                predEvals = []
                numEvalsOption = evalOptions["numEvals"]

                # SAMPLE SEQUENTIALLY
                if numEvalsOption == 1 or evalOptions["sequentialEvalRuns"][shortTestSetName]:
                    print("\tSampling sequentially for %d samples..." % (numEvalsOption))

                    timerEvalsStart = time.perf_counter()
                    for run in range(numEvalsOption):
                        predSamples = []
                        for s, sample in enumerate(testLoader, 0):
                            dataPath = sample["path"]
                            data = sample["data"].to(device)
                            simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None

                            prediction, _, _ = model(data, simParameters)
                            prediction = prediction.unsqueeze(0).unsqueeze(0)
                            predSamples += [prediction.cpu().numpy()]

                        predSamples = np.concatenate(predSamples, axis=2)
                        if numEvalsOption == 1: # repeat same evalation result for non-probabilistic models
                            predSamples = np.repeat(predSamples, numEvals, axis=1)

                        # undo data normalization
                        normMean = testTransformations.normMean[dataTransformationFieldIndices]
                        normStd = testTransformations.normStd[dataTransformationFieldIndices]
                        normMean = np.expand_dims(normMean, axis=(0,1,2,3,5,6))
                        normStd = np.expand_dims(normStd, axis=(0,1,2,3,5,6))
                        predSamples = (predSamples * normStd) + normMean

                        predEvals += [predSamples]

                        timerEvalsEnd = time.perf_counter()
                        print("\t[%2.2f min] Sample %d done." % ((timerEvalsEnd - timerEvalsStart)/60, run))

                # SAMPLE IN PARALLEL
                else:
                    predSamples = []
                    print("\tSampling in parallel for %d samples..." % (numEvalsOption))

                    timerEvalsStart = time.perf_counter()
                    for s, sample in enumerate(testLoader, 0):
                        dataPath = sample["path"]
                        data = sample["data"].to(device).repeat(numEvalsOption,1,1,1,1)

                        if type(sample["simParameters"]) is not dict:
                            simParameters = sample["simParameters"].to(device)
                            simParameters = simParameters.repeat(numEvalsOption,1,1)
                        else:
                            simParameters = None

                        prediction, _, _ = model(data, simParameters)

                        prediction = torch.reshape(prediction, (1, numEvalsOption, -1, data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
                        predSamples += [prediction.cpu().numpy()]

                    predSamples = np.concatenate(predSamples, axis=2)

                    # undo data normalization
                    normMean = testTransformations.normMean[dataTransformationFieldIndices]
                    normStd = testTransformations.normStd[dataTransformationFieldIndices]
                    normMean = np.expand_dims(normMean, axis=(0,1,2,3,5,6))
                    normStd = np.expand_dims(normStd, axis=(0,1,2,3,5,6))
                    predSamples = (predSamples * normStd) + normMean

                    predEvals += [predSamples]

                    timerEvalsEnd = time.perf_counter()
                    print("\t[%2.2f min] All samples done." % ((timerEvalsEnd - timerEvalsStart)/60))

                predEvals = np.concatenate(predEvals, axis=1)
                predFull += [predEvals]

        predFull = np.concatenate(predFull, axis=0)
        np.savez_compressed(os.path.join(testSetOutPath, modelName + ".npz"), predFull)
        timerFullEnd = time.perf_counter()

        print("\n[%2.2f min] Saved prediction to %s with shape %s\n" % ((timerFullEnd-timerFullStart)/60, os.path.join(testSetOutPath, modelName + ".npz"), predFull.shape))
        # output shape:
        # [trained models,  number of evaluations,  data set sequences,  sequence length,  channels,  positionX,  positionY]

print("\n\nSampling complete.")

