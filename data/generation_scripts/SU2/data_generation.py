import numpy as np
import os, shutil, sys, atexit
from multiprocessing import Process
from convert_data import convertFolderToNumpy, watchAndConvertFolder, createObstacleMask

baseFolder = "SU2_raw"
SU2path = "../build/bin/SU2_CFD"               # relative to baseFolder
cfgNameSteady = "steady.cfg"                   # must be contained in baseFolder
cfgNameInitial = "unsteady_2d_initial.cfg"
cfgNameMain = "unsteady_2d_lowDissipation.cfg"
linkFiles = ["grid_quad_2d.su2"]               # must be contained in baseFolder, will only be symlinked to simfolders
threads = int(sys.argv[1])                           if len(sys.argv) > 1 else 112

reynolds = int(sys.argv[2])                          if len(sys.argv) > 2 else 3900
machs = [float(i) for i in sys.argv[3].split(",")]   if len(sys.argv) > 3 else [0.2]
simId = [int(i) for i in sys.argv[4].split(",")]     if len(sys.argv) > 4 else [0]
restartIter = int(sys.argv[5])                       if len(sys.argv) > 5 else -1

convertOnline = True
reduceFrameNum = (100000, 50)
res = (256j,128j,1j)
resampleGrid = np.mgrid[-1.5:10.5:res[0], -3:3:res[1]] # unsteady cyl domain: x -31:31  y -31:31

assert(len(machs) == len(simId)), "Number of simulations does not match number of mach numbers!"

# read cfg files
with open(os.path.join(baseFolder, cfgNameSteady), "r") as f:
    cfgSteady = f.read()
    f.close()
with open(os.path.join(baseFolder, cfgNameInitial), "r") as f:
    cfgInitial = f.read()
    f.close()
with open(os.path.join(baseFolder, cfgNameMain), "r") as f:
    cfgMain = f.read()
    f.close()


# folder and file management
os.chdir(baseFolder)
for id, mach in zip(simId, machs):
    simDir = "sim_%06d" % (id)
    if os.path.isdir(simDir):
        shutil.rmtree(simDir)
    os.mkdir(simDir)

    # update generic cfg templates and copy to sim folder
    for cfgName, cfg in zip([cfgNameSteady, cfgNameInitial, cfgNameMain], [cfgSteady, cfgInitial, cfgMain]):
        with open(os.path.join(simDir, cfgName), "w") as f:
            oldRey = "REYNOLDS_NUMBER= #GEN_REPLACE"
            newRey = "REYNOLDS_NUMBER= %1.1f" % (reynolds)
            updatedCfg = cfg.replace(oldRey, newRey)

            oldMach = "MACH_NUMBER= #GEN_REPLACE"
            newMach = "MACH_NUMBER= %1.4f" % (mach)
            updatedCfg = updatedCfg.replace(oldMach, newMach)

            oldDt = "TIME_STEP= #GEN_REPLACE"
            newDt = "TIME_STEP= %1.10f" % (0.0000293861 * (0.2 / mach))  # 0.0000293861 is reference timestep for mach 0.2, adjust linearly to different mach
            updatedCfg = updatedCfg.replace(oldDt, newDt)

            if restartIter > 0 and cfgName == cfgMain:
                updatedCfg = updatedCfg.replace("RESTART_ITER= 100000", "RESTART_ITER= %d" % restartIter)
                if restartIter > 100000:
                    updatedCfg = updatedCfg.replace("READ_BINARY_RESTART= YES", "READ_BINARY_RESTART= NO")

            f.write(updatedCfg)
            f.close()

    # symlink large misc files like meshes to sim folder to avoid copies
    os.chdir(simDir)
    for linkFile in linkFiles:
        os.symlink(os.path.join("..", linkFile), linkFile)

    # create obstacle mask
    createObstacleMask(".", resampleGrid, 1.0)

    ### SETUP FOR MAIN SIMULATION
    if restartIter < 0:
        ### STEADY SIMULATION FOR 1000 STEPS TO INITIALIZE FLOW FIELD
        # call SU2 solver
        result = os.system("mpiexec -n %d ../%s %s" % (threads, SU2path, cfgNameSteady))
        if result != 0:
            raise RuntimeError("Steady SU2 process failed! %s" % str(result))

        # prepare files for restarting
        shutil.copy("restart_flow.dat", "solution_flow_00998.dat")
        os.rename("restart_flow.dat", "solution_flow_00999.dat")


        ### UNSTEADY SIMULATION FOR 100000 STEPS TO PASS TRANSIENT STAGE
        # call SU2 solver
        result = os.system("mpiexec -n %d ../%s %s" % (threads, SU2path, cfgNameInitial))
        if result != 0:
            raise RuntimeError("Unsteady initial SU2 process failed! %s" % str(result))

        # prepare files for restarting
        if not os.path.isdir("../restart"):
            os.mkdir("../restart")
        shutil.copy("restart_flow_99998.dat", "../restart/rey%06d_mach%1.4f_sol_99998.dat" % (reynolds, mach))
        shutil.copy("restart_flow_99999.dat", "../restart/rey%06d_mach%1.4f_sol_99999.dat" % (reynolds, mach))
        os.rename("restart_flow_99998.dat", "solution_flow_99998.dat")
        os.rename("restart_flow_99999.dat", "solution_flow_99999.dat")

    ### DIRECT RESTART OF MAIN SIMULATION
    else:
        fileType = "dat" if restartIter <= 100000 else "csv"
        shutil.copy("../restart/rey%06d_mach%1.4f_sol_%05d.%s" % (reynolds, mach, restartIter-2, fileType), "solution_flow_%05d.%s" % (restartIter-2, fileType))
        shutil.copy("../restart/rey%06d_mach%1.4f_sol_%05d.%s" % (reynolds, mach, restartIter-1, fileType), "solution_flow_%05d.%s" % (restartIter-1, fileType))


    ### MAIN SIMULATION TO GENERATE DATA
    # setup conversion in separate process via folder watcher
    if convertOnline:
        print("[CONVERSION WATCHER] Started conversion watcher for: %s" % simDir)
        convertWatcher = Process(target=watchAndConvertFolder, args=(1, ".", resampleGrid, reduceFrameNum, 2))
        convertWatcher.start()
        atexit.register(lambda proc : proc.terminate(), convertWatcher)

    # call SU2 solver
    result = os.system("mpiexec -n %d ../%s %s" % (threads, SU2path, cfgNameMain))
    if result != 0:
        raise RuntimeError("Main SU2 process failed! %s" % str(result))

    # terminate watcher and finalize conversion
    if convertOnline:
        convertWatcher.terminate()
        convertFolderToNumpy(".", resampleGrid, reduceFrameNum, 0)
        print("[CONVERSION WATCHER] Terminated conversion watcher for: %s" % simDir)

    os.chdir("..")
    print("\n\n\n\n")

print("\n\n\n\n\nSIMULATIONS COMPLETE!")
