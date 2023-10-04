import numpy as np
import os, time, csv, re
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt


def convertFolderToNumpy(folderPath:str, resampleGrid:np.ndarray, reduceFrameNum:tuple, ignoreLatest:int):
    visualize = False # plot quick visualization of converted files
    dimension = 2 # 2 = 2D data and  3 = 3D data
    ignoreZ = False # do not save Z dimension if 2D data has a z dimension of 1

    grid = resampleGrid.astype(np.float32)

    #print("[CONVERSION WATCHER] Converting folder to numpy...")
    files = os.listdir(folderPath)
    files.sort(reverse=True)

    ignored = ignoreLatest
    for fileName in files:
        filePath = os.path.join(folderPath, fileName)

        if "restart_flow" in os.path.splitext(fileName)[0] and os.path.splitext(fileName)[1] == ".csv":
            if ignored > 0:
                ignored -= 1
                continue

            reMatch = re.search("[0-9]+", fileName)
            if not reMatch:
                raise ValueError("Invalid frame filename!")
            frameNr = int(reMatch.group())
            if (frameNr- reduceFrameNum[0]) % reduceFrameNum[1] != 0:
                os.remove(filePath)
                continue
            frameNr = int((frameNr - reduceFrameNum[0]) / reduceFrameNum[1])


            with open(filePath, "r") as csvFile:
                reader = csv.DictReader(csvFile)

                points = {"x" : [], "y" : []}
                values = {"Density" : [], "Pressure" : [], "Momentum_x" : [], "Momentum_y" : []}
                if dimension == 3:
                    points["z"] = []
                    values["Momentum_z"] = []

                for col in reader:
                    for poi,list in points.items():
                        list.append(col[poi])
                    for val,list in values.items():
                        list.append(col[val])

                csvFile.close()

                pos = []
                for _,list in points.items():
                    pos.append(np.array(list, dtype=np.float32))
                pos = np.stack(pos, axis=1)
                channels = []
                for _,list in values.items():
                    channels.append(np.array(list, dtype=np.float32))
                channels = np.stack(channels, axis=1)

                if dimension == 2:
                    #resampled = griddata(pos, channels, (resampleGrid[0], resampleGrid[1]), method="cubic")
                    rbf = RBFInterpolator(pos, channels, neighbors=5, kernel="linear")
                    resampled = rbf(grid.reshape(2, -1).T)
                    resampled = resampled.reshape(grid.shape[1], grid.shape[2], channels.shape[1])
                else:
                    rbf = RBFInterpolator(pos, channels, neighbors=5, kernel="linear")
                    resampled = rbf(grid.reshape(3, -1).T)
                    resampled = resampled.reshape(grid.shape[1], grid.shape[2], grid.shape[3], channels.shape[1])

                if visualize:
                    # quick visualization
                    if dimension == 2:
                        r = np.transpose(resampled, (1,0,2))
                        plt.subplot(221)
                        plt.title("Density")
                        plt.imshow(r[...,0], interpolation="nearest")
                        plt.subplot(222)
                        plt.title("Pressure")
                        plt.imshow(r[...,1], interpolation="nearest")
                        plt.subplot(223)
                        plt.title("Velocity X")
                        plt.imshow(r[...,2], interpolation="nearest")
                        plt.subplot(224)
                        plt.title("Velocity Y")
                        plt.imshow(r[...,3], interpolation="nearest")
                    elif dimension == 3:
                        r = np.mean(resampled, axis=2)
                        r = np.transpose(r, (1,0,2))
                        plt.subplot(231)
                        plt.title("Density")
                        plt.imshow(r[...,0], interpolation="nearest")
                        plt.subplot(232)
                        plt.title("Pressure")
                        plt.imshow(r[...,1], interpolation="nearest")
                        plt.subplot(233)
                        plt.title("Velocity X")
                        plt.imshow(r[...,2], interpolation="nearest")
                        plt.subplot(234)
                        plt.title("Velocity Y")
                        plt.imshow(r[...,3], interpolation="nearest")
                        plt.subplot(235)
                        plt.title("Velocity Z")
                        plt.imshow(r[...,4], interpolation="nearest")
                    plt.savefig(os.path.join(folderPath, "vis_%06d" % frameNr), bbox_inches="tight", dpi=400)
                    plt.close("all")

                resampled = np.transpose(resampled, (2,0,1)) if dimension == 2 else np.transpose(resampled, (3,0,1,2))
                if ignoreZ:
                    resampled = np.squeeze(resampled[0:5], axis=3)

                dens = resampled[0:1]
                densPath = os.path.join(folderPath, "density_%06d" % frameNr)
                np.savez_compressed(densPath, dens)

                pres = resampled[1:2]
                presPath = os.path.join(folderPath, "pressure_%06d" % frameNr)
                np.savez_compressed(presPath, pres)

                vel = resampled[2:4] if dimension == 2 else resampled[2:5]
                velPath = os.path.join(folderPath, "velocity_%06d" % frameNr)
                np.savez_compressed(velPath, vel)

                os.remove(filePath)



def watchAndConvertFolder(watchInterval:int, folderPath:str, resampleGrid:np.ndarray, reduceFrameNumMod:int, ignoreLatest:int):
    while True:
        time.sleep(watchInterval)
        convertFolderToNumpy(folderPath, resampleGrid, reduceFrameNumMod, ignoreLatest)



def createObstacleMask(folderPath:str, resampleGrid:np.ndarray, diameter:float):
    if not os.path.isfile(os.path.join(folderPath, "obstacle_mask.npz")):
        grid = resampleGrid.astype(np.float32)
        radius = float(diameter) / 2
        mask = grid[0] * grid[0] + grid[1] * grid[1] > radius * radius
        mask = mask.astype(int)

        maskPath = os.path.join(folderPath, "obstacle_mask.npz")
        np.savez_compressed(maskPath, mask)
