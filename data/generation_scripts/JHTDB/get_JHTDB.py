import numpy as np
import os, json

from pyJHTDB import libJHTDB
import pyJHTDB.dbinfo as dbinfo

outDir = "data/128_iso/"
dataset = dbinfo.isotropic1024coarse
zSlices = list(range(0, 1000, 1))
size = (128, 64)
strideSpatial = (2, 2)
timeSteps = range(0, 1001, 1)


def download(outDir:str, dataset:dict, zSlices:list, size:tuple, strideSpatial:tuple, timeSteps:list):
    lJHTDB = libJHTDB()
    lJHTDB.initialize()
    lJHTDB.add_token("ADD YOUR TOKEN HERE")

    assert (dataset["nx"] >= size[0] * strideSpatial[0] and size[0] >= 1), "Incorrect x size settings"
    assert (dataset["ny"] >= size[1] * strideSpatial[1] and size[1] >= 1), "Incorrect y size settings"
    assert ((dataset["nz"] >= z+1 and z+1 >= 1) for z in zSlices), "Incorrect z slicing"
    assert (dataset["time"].shape[0] >= timeSteps[-1]+1 and timeSteps[0]+1 >= 1), "Incorrect time settings"

    print("Starting download...")
    start = np.array( [1, 1, zSlices[0]+1] ).astype(np.int32)
    end = np.array( [size[0]*strideSpatial[0], size[1]*strideSpatial[1], zSlices[-1]+2] ).astype(np.int32)
    step = np.array( [strideSpatial[0], strideSpatial[1], 1] ).astype(np.int32)

    # create folder structure
    for z in zSlices:
        outPath = os.path.join(outDir, "sim_%06d" % (z))
        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        srcPath = os.path.join(outPath, "src")
        if not os.path.isdir(srcPath):
            os.makedirs(srcPath)

    # main download
    for t in timeSteps:
        # velocity
        outFileVel = os.path.join(outDir, "sim_%06d" % (zSlices[-1]), "velocity_%06d.npz" % (t))
        outFileVelZ = os.path.join(outDir, "sim_%06d" % (zSlices[-1]), "velocityZ_%06d.npz" % (t))

        if not os.path.isfile(outFileVel) or not os.path.isfile(outFileVelZ):
            vel = lJHTDB.getCutout( data_set=dataset["name"], field='u', time_step=t+1, start=start, end=end, step=step).astype(np.float32)
            vel = np.transpose(vel, axes=[0,3,2,1])
            print("VELOCITY - Time: %d, start: (%d %d %d), end: (%d %d %d), step: (%d %d %d)" % (t+1, start[0], start[1], start[2], end[0], end[1], end[2], step[0], step[1], step[2]))
            print("\t" + str(vel.shape))

            for z in zSlices:
                outPath = os.path.join(outDir, "sim_%06d" % (z))

                outFile = os.path.join(outPath, "velocity_%06d.npz" % (t))
                if not os.path.isfile(outFile):
                    data = vel[z - zSlices[0], 0:2] # ignore z velocity component
                    np.savez_compressed(outFile, data)

                outFile = os.path.join(outPath, "velocityZ_%06d.npz" % (t))
                if not os.path.isfile(outFile):
                    data = vel[z - zSlices[0], 2:3] # ignore x and y velocity component
                    np.savez_compressed(outFile, data)

        else:
            print("VELOCITY - Time: %d files already exist, skipping..." % (t))


        # pressure
        outFilePres = os.path.join(outDir, "sim_%06d" % (zSlices[-1]), "pressure_%06d.npz" % (t))
        if not os.path.isfile(outFilePres):
            pres = lJHTDB.getCutout( data_set=dataset["name"], field='p', time_step=t+1, start=start, end=end, step=step).astype(np.float32)
            pres = np.transpose(pres, axes=[0,3,2,1])
            print("PRESSURE - Time: %d, start: (%d %d %d), end: (%d %d %d), step: (%d %d %d)" % (t+1, start[0], start[1], start[2], end[0], end[1], end[2], step[0], step[1], step[2]))
            print("\t" + str(pres.shape))

            for z in zSlices:
                outPath = os.path.join(outDir, "sim_%06d" % (z))

                outFile = os.path.join(outPath, "pressure_%06d.npz" % (t))
                if not os.path.isfile(outFile):
                    data = pres[z - zSlices[0]]
                    np.savez_compressed(outFile, data)

        else:
            print("PRESSURE - Time: %d files already exist, skipping..." % (t))

    # logs
    for z in zSlices:
        start = np.array( [1, 1, z+1] ).astype(np.int32)
        end = np.array( [size[0]*strideSpatial[0], size[1]*strideSpatial[1], z+2] ).astype(np.int32)
        step = np.array( [strideSpatial[0], strideSpatial[1], 1] ).astype(np.int32)

        log = {}
        log["JHTDB Dataset"] = dataset["name"]
        log["Z Slice"] = z+1
        log["Resolution"] = [size[0], size[1]]
        log["Fields"] = ["Velocity X", "Velocity Y", "Velocity Z", "Pressure"]
        log["Cutout Positions"] = []

        for t in timeSteps:
            log["Cutout Positions"].append("idx: %d, timestep: %d, start: (%d %d %d), end: (%d %d %d), step: (%d %d %d)" % (t, t+1, start[0], start[1], start[2], end[0], end[1], end[2], step[0], step[1], step[2]))

        logFile = os.path.join(outDir, "sim_%06d" % (z), "src", "description.json")
        with open(logFile, 'w') as f:
            json.dump(log, f, indent=4)
            f.close()

    lJHTDB.finalize()
    print("\nDownload finished")


if __name__ == '__main__':
    download(outDir, dataset, zSlices, size, strideSpatial, timeSteps) #type:ignore