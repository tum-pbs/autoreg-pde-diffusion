# conversion of SU2 directory structure and parsing of values like mach and reynolds numbers, and lift and drag coefficients

import numpy as np
import os, shutil, csv, re, json
import imageio, matplotlib


dimension = 2
ignoreZ = False
inDir = "data/SU2_raw/"
outDir = "data/256_tra/"
cfgName = "unsteady_2d_lowDissipation.cfg"
historyName = "history_100000.csv"
historyReduceFrameNum = (100000, 50)

render = False

for root, dirs, files in os.walk(inDir):
    # copy top level files
    if root == inDir:
        for fileName in files:
            filePath = os.path.join(root, fileName)
            outPath = filePath.replace(inDir, outDir)
            outFolder = os.path.dirname(outPath)
            if not os.path.exists(outFolder):
                os.makedirs(outFolder)
            shutil.copy(filePath, outPath)

    # handle individual simulation folders
    else:
        descriptionJson = {}
        for fileName in sorted(files):
            filePath = os.path.join(root, fileName)
            outPath = filePath.replace(inDir, outDir)
            outFolder = os.path.dirname(outPath)
            outFolderSrc = os.path.join(outFolder, "src")
            outPathSrc = os.path.join(outFolderSrc, fileName)

            if "restart" in outFolder or "__pycache__" in outFolder:
                continue

            if not os.path.exists(outFolder):
                os.makedirs(outFolder)
            if not os.path.exists(outFolderSrc):
                os.makedirs(outFolderSrc)

            # copy auxiliary files to src folder
            if os.path.splitext(fileName)[1] in [".cfg"] or any(n in os.path.splitext(fileName)[0] for n in ["history","restart_flow","solution_flow"]):
                shutil.copy(filePath, outPathSrc)

                # parse reynolds/mach number from cfg file
                if fileName == cfgName:
                    with open(filePath, "r") as cfgFile:
                        cfgStr = cfgFile.read()
                        reMatch = re.search("REYNOLDS_NUMBER= [0-9]+[.]?[0-9]*", cfgStr)
                        if not reMatch:
                            raise ValueError("Could not parse reynolds number!")
                        reMatch = re.search("[0-9]+[.]?[0-9]*", reMatch.group())
                        reynolds = float(reMatch.group())
                        descriptionJson["Reynolds Number"] = reynolds

                        maMatch = re.search("MACH_NUMBER= [0-9]+[.]?[0-9]*", cfgStr)
                        if not maMatch:
                            raise ValueError("Could not parse mach number!")
                        maMatch = re.search("[0-9]+[.]?[0-9]*", maMatch.group())
                        mach = float(maMatch.group())
                        descriptionJson["Mach Number"] = mach
                    cfgFile.close()

                # parse lift/drag coefficients from history file
                if fileName == historyName:
                    with open(filePath, "r") as csvFile:
                        csvReader = csv.DictReader(csvFile, doublequote=True)
                        coefDrag = []
                        coefLift = []
                        for row in csvReader:
                            stripRow = {}
                            for key, value in row.items():
                                stripRow[key.strip().replace("\"","")] = value.strip()

                            timeIter = int(stripRow["Time_Iter"])
                            if (timeIter - historyReduceFrameNum[0]) % historyReduceFrameNum[1] == 0:
                                coefDrag += [float(stripRow["CD"])]
                                coefLift += [float(stripRow["CL"])]
                    csvFile.close()
                    descriptionJson["Drag Coefficient"] = coefDrag
                    descriptionJson["Lift Coefficient"] = coefLift

            # copy simulation npz files
            elif os.path.splitext(fileName)[1] == ".npz":
                shutil.copy(filePath, outPath)

        # write simulation description json
        if descriptionJson:
            jsonPath = os.path.join(outFolderSrc, "description.json")
            with open(jsonPath, 'w') as jsonFile:
                json.dump(descriptionJson, jsonFile, indent=4)
            jsonFile.close()


# collect the npz sequences and render them into mp4 files
if render:
    print("\nStarting video rendering...")
    useDens = True

    simDirs = os.listdir(outDir)
    simDirs.sort()
    for simDir in simDirs:
        simDir = os.path.join(outDir, simDir)
        if not os.path.isdir(simDir):
            continue

        print("Rendering %s" % (simDir))
        simFiles = os.listdir(simDir)
        simFiles.sort()

        minFrame = float("inf")
        maxFrame = float("-inf")

        for simFile in simFiles:
            if os.path.isdir(os.path.join(simDir, simFile)):
                continue

            reMatch = re.search("[0-9]+", simFile)
            if not reMatch:
                continue
            frameNr = int(reMatch.group())
            if frameNr < minFrame:
                minFrame = frameNr
            if frameNr > maxFrame:
                maxFrame = frameNr

        recVel = []
        recPres = []
        recDens = []
        for i in range(minFrame, maxFrame+1):
            pres = os.path.join(simDir, "pressure_%06d.npz" % i)
            vel = os.path.join(simDir, "velocity_%06d.npz" % i)
            dens = os.path.join(simDir, "density_%06d.npz" % i)
            densOk = useDens and os.path.isfile(dens)
            if os.path.isfile(pres) and os.path.isfile(vel) and densOk:
                if dimension == 2 or ignoreZ:
                    recPres += [np.load(pres)["arr_0"]]
                    recVel += [np.load(vel)["arr_0"]]
                    if densOk:
                        recDens += [np.load(dens)["arr_0"]]
                else:
                    recPres += [np.mean(np.load(pres)["arr_0"], axis=3)]
                    recVel += [np.mean(np.load(vel)["arr_0"], axis=3)]
                    if densOk:
                        recDens += [np.mean(np.load(dens)["arr_0"], axis=3)]
            else:
                raise ValueError("Missing file: %s %s %s" % (pres, vel, dens))

        recVel = np.transpose(np.stack(recVel, axis=0), axes=[0,2,3,1])
        recPres = np.transpose(np.stack(recPres, axis=0), axes=[0,2,3,1])
        if useDens:
            recDens = np.transpose(np.stack(recDens, axis=0), axes=[0,2,3,1])

        desc = os.path.join(simDir, "src", "description.json")
        if not os.path.isfile(desc):
            raise ValueError("Missing src file: %s" % desc)
        with open(desc) as f:
            loaded = json.load(f)
            reynolds = loaded["Reynolds Number"]
            mach = loaded["Mach Number"]
            f.close()


        # rendering
        renderpath = os.path.join(simDir, "render")
        if not os.path.exists(renderpath):
            os.makedirs(renderpath)
        renderfile = "rey%06d_mach%1.4f.mp4" % (reynolds, mach)

        # NOT identical to PhiFlow rendering in the following (added density)
        vx_dx, vx_dy = np.gradient(recVel[...,0][...,None], axis=(1,2))
        vy_dx, vy_dy = np.gradient(recVel[...,1][...,None], axis=(1,2))
        curl = vy_dx - vx_dy
        divergence = vx_dx + vy_dy

        if useDens:
            renderdata = [[recVel[...,0][...,None],curl], [recVel[...,1][...,None],recDens], [recVel,recPres]]
            rendercmap = [["seismic","seismic"], ["seismic","Blues"], [None,"PuOr"]]
        else:
            renderdata = [[recVel[...,0][...,None],curl], [recVel[...,1][...,None],divergence], [recVel,recPres]]
            rendercmap = [["seismic","seismic"], ["seismic","coolwarm"], [None,"PuOr"]]

        pad = 8
        result = []
        for i in range(len(renderdata)):
            rows = []
            for j in range(len(renderdata[i])):
                part = np.copy(renderdata[i][j])
                part = np.rot90(part, axes=(1,2))
                cmap = rendercmap[i][j]
                if cmap:
                    cmap = matplotlib.cm.get_cmap(cmap)

                for k in range(part.shape[-1]):
                    pMax = max(abs(np.min(part[...,k])), abs(np.max(part[...,k])))
                    pMin = -pMax
                    #pMax = np.max(part[...,k])
                    #pMin = np.min(part[...,k])
                    part[...,k] = (part[...,k] - pMin) / (pMax - pMin)

                if part.shape[-1] == 1 and cmap:
                    part = cmap(np.squeeze(part))

                if part.shape[-1] == 2:
                    blue = np.zeros((part.shape[0], part.shape[1], part.shape[2], 1))
                    alpha = np.ones_like(blue)
                    part = np.concatenate([part, blue, alpha], axis=3)

                if part.shape[-1] == 3:
                    alpha = np.ones((part.shape[0], part.shape[1], part.shape[2], 1))
                    part = np.concatenate([part, alpha], axis=3)

                part = 255 * np.pad(part, ((0,0), (pad,pad), (pad,pad), (0,0)) )
                rows += [part.astype(np.uint8)]
            result += [np.concatenate(rows, axis=1)]
        result = np.concatenate(result, axis=2)

        vidfile = renderfile
        imageio.mimwrite(os.path.join(renderpath, vidfile), result, quality=10, fps=5, ffmpeg_log_level="error")