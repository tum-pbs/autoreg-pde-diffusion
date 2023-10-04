# conversion of simulation data to lower resolutions

import numpy as np
import os, shutil
import scipy.ndimage

inDir = "data/256_inc/"
outDir = "data/128_inc/"
#inDir = "data/256_reyVar/"
#outDir = "data/128_reyVar/"

#inDir = "data/256_tra/"
#outDir = "data/128_tra/"
res = (128,64)

for root, dirs, files in os.walk(inDir):
    for fileName in files:
        filePath = os.path.join(root, fileName)
        outPath = filePath.replace(inDir, outDir)
        outFolder = os.path.dirname(outPath)

        if not os.path.exists(outFolder):
            os.makedirs(outFolder)

        if os.path.splitext(fileName)[1] == ".npz":
            data = np.load(filePath)['arr_0']

            order = 3
            if data.ndim == 3:
                zoom = [1, res[0]/data.shape[1], res[1]/data.shape[2]]
            elif data.ndim == 4:
                zoom = [1, res[0]/data.shape[1], res[1]/data.shape[2], res[2]/data.shape[3]]
            elif data.ndim == 2 and "obstacle_mask" in os.path.splitext(fileName)[0]:
                order = 0
                zoom = [res[0]/data.shape[0], res[1]/data.shape[1]]
            else:
                raise ValueError("Invalid data dimensions: %s %s" % (fileName, str(data.shape)))

            dataLow = scipy.ndimage.zoom(data, zoom, order=order, grid_mode=True)

            np.savez_compressed(outPath, dataLow)
            print("%s  %s  %s" % (filePath,str(data.shape),str(dataLow.shape)))

        else:
            shutil.copy(filePath, outPath)

