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
modelFolder = "models/models_tra"

testSets = {
    "extrap":
        TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", ["data"], filterTop=["128_tra"], filterSim=[(0,3)],
                filterFrame=[(500,750)], sequenceLength=[[60,2]], simFields=["dens","pres"], simParams=["mach"], printLevel="sim"),
    "interp" :
        TurbulenceDataset("Test Interpolate Mach 0.66-0.68", ["data"], filterTop=["128_tra"], filterSim=[(16,19)],
                filterFrame=[(500,750)], sequenceLength=[[60,2]], simFields=["dens","pres"], simParams=["mach"], printLevel="sim"),
    "longer" :
        TurbulenceDataset("Test Longer Rollout Mach 0.64-0.65", ["data"], filterTop=["128_tra"], filterSim=[(14,16)],
                filterFrame=[(0,1000)], sequenceLength=[[240,2]], simFields=["dens","pres"], simParams=["mach"], printLevel="sim"),
}
dataTransformationFieldIndices = [0,1,2,3,5] # ORDER (fields): velocity (x,y), velocity z / density, pressure, ORDER (params): rey, mach, zslice

numEvals = 5

models = {
    "tf-enc":
        (
            [
            "128_tf-enc_00/Model.pth",
            "128_tf-enc_01/Model.pth",
            "128_tf-enc_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),

    "tf-vae":
        (
            [
            "128_tf-vae_00/Model.pth",
            "128_tf-vae_01/Model.pth",
            "128_tf-vae_02/Model.pth",
            ],
            {"extrap": 4, "interp": 4, "longer": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": False, "interp": False, "longer": True}
            },
        ),

    "tf-mgn":
        (
            [
            "128_tf-mgn_00/Model.pth",
            "128_tf-mgn_01/Model.pth",
            "128_tf-mgn_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),


    "unet-m2":
        (
            [
            "128_unet-m2_00/Model.pth",
            "128_unet-m2_01/Model.pth",
            "128_unet-m2_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),

    "unet-m8":
        (
            [
            "128_unet-m8_00/Model.pth",
            "128_unet-m8_01/Model.pth",
            "128_unet-m8_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),

    "unet-m2-noise0.01":
        (
            [
            "128_unet-m2-noise0.01_00/Model.pth",
            "128_unet-m2-noise0.01_01/Model.pth",
            "128_unet-m2-noise0.01_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),


    "refiner-r4_std0.000001":
       (
            [
            "128_refiner-r4_std0.000001_00/Model.pth",
            "128_refiner-r4_std0.000001_01/Model.pth",
            "128_refiner-r4_std0.000001_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
           },
       ),


    "resnet-m2":
        (
            [
            "128_resnet-m2_00/Model.pth",
            "128_resnet-m2_01/Model.pth",
            "128_resnet-m2_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),

    "dil-resnet-m2":
        (
            [
            "128_dil-resnet-m2_00/Model.pth",
            "128_dil-resnet-m2_01/Model.pth",
            "128_dil-resnet-m2_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),


    "fno-16modes-m2":
        (
            [
            "128_fno-16modes-m2_00/Model.pth",
            "128_fno-16modes-m2_01/Model.pth",
            "128_fno-16modes-m2_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),

    "fno-32modes-m2":
        (
            [
            "128_fno-32modes-m2_00/Model.pth",
            "128_fno-32modes-m2_01/Model.pth",
            "128_fno-32modes-m2_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 1},
            {"numEvals": 1,
            },
        ),


    "acdm-r20_ncn":
        (
            [
            "128_acdm-r20_ncn_00/Model.pth",
            "128_acdm-r20_ncn_01/Model.pth",
            "128_acdm-r20_ncn_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "clean"
            },
        ),

    "acdm-r20":
        (
            [
            "128_acdm-r20_00/Model.pth",
            "128_acdm-r20_01/Model.pth",
            "128_acdm-r20_02/Model.pth",
            ],
            {"extrap": 8, "interp": 8, "longer": 4},
            {"numEvals": numEvals,
            "sequentialEvalRuns": {"extrap": True, "interp": True, "longer": True},
            "samplingMode": "ddpm",
            "posteriorSampling": "random",
            "initialSampling": "random",
            "conditioningIntegration": "noisy"
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
            model = PredictionModel.load(os.path.join(modelFolder, modelPath), useGPU=device=="cuda")
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