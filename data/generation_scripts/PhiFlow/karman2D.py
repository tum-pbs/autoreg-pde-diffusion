""" Karman vortex street
Simulates a viscous fluid flowing around a cylinder.
"""

import os, sys, json, time, datetime
import imageio, matplotlib
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from phi.torch.flow import *
phi.torch.TORCH.set_default_device("GPU")


dataDir = "data/256_inc"
write = True
readOnly, readIdx = False, 0
render = False
writeImageSequence = False

### DEFAULT SIMULATION PARAMETERS
RES_X, RES_Y = 256, 128
DT = 0.05
STEPS, WARMUP = 1300, 20

CYL_SIZE = 0.6
WALL_TOP, WALL_BOTTOM = (7/6)*CYL_SIZE, (7/6)*CYL_SIZE
WALL_LEFT, WALL_RIGHT = (7/6)*CYL_SIZE, 4.5*CYL_SIZE
VEL_IN = 0.5
VISC_START = 0.0005
VISC_END = 0.0005

#VEL = VEL_IN * CYL_SIZE # use when changing cyl size
VEL = VEL_IN
REYNOLDS_START = (VEL * CYL_SIZE) / VISC_START
REYNOLDS_END = (VEL * CYL_SIZE) / VISC_END


### ARGUMENT PARSING
gui = "console"
#gui = "dash"
if len(sys.argv) == 12:
    dataDir = "data/%s" % sys.argv[1]
    RES_X = int(sys.argv[2])
    RES_Y = int(sys.argv[3])
    DT = float(sys.argv[4])
    STEPS = int(sys.argv[5])
    WARMUP = int(sys.argv[6])
    CYL_SIZE = float(sys.argv[7])
    VEL_IN = float(sys.argv[8])

    WALL_TOP, WALL_BOTTOM = (7/6)*CYL_SIZE, (7/6)*CYL_SIZE
    WALL_LEFT, WALL_RIGHT = (7/6)*CYL_SIZE, 4.5*CYL_SIZE

    if sys.argv[9] != "-":
        VISC_START = float(sys.argv[9])
        VISC_END = float(sys.argv[9])
        #VEL = VEL_IN * CYL_SIZE
        VEL = VEL_IN
        REYNOLDS_START = (VEL * CYL_SIZE) / VISC_START
        REYNOLDS_END = (VEL * CYL_SIZE) / VISC_END

    if sys.argv[10] != "-" and sys.argv[11] != "-":
        REYNOLDS_START = float(sys.argv[10])
        REYNOLDS_END = float(sys.argv[11])
        #VEL = VEL_IN * CYL_SIZE
        VEL = VEL_IN
        VISC_START = (VEL * CYL_SIZE) / REYNOLDS_START
        VISC_END = (VEL * CYL_SIZE) / REYNOLDS_END

    gui = "console"
else:
    print("WARNING: No parameter arguments!")
    #os._exit(-1)


print("--------------------------------------------")
print("| Resolution: (%d, %d)" % (RES_X, RES_Y))
print("| Dt: %1.3f" % (DT))
print("| Steps (Warmup): %d (%d)" % (STEPS, WARMUP))
print("| Cylinder Size: %1.3f" % (CYL_SIZE))
print("| Inflow Velocity: %1.3f" % (VEL))
print("| Fluid Viscosity: (%1.8f, %1.8f)" % (VISC_START, VISC_END))
print("| REYNOLDS NUMBER: (%d, %d)" % (REYNOLDS_START, REYNOLDS_END))
print("--------------------------------------------\n")
#os._exit(-1)



### SCENE SETUP
scene = Scene.create(dataDir) if not readOnly else Scene.at(dataDir, readIdx)

DOMAIN = dict(x=RES_X, y=RES_Y, bounds=Box[0:WALL_LEFT + CYL_SIZE + WALL_RIGHT, 0:WALL_BOTTOM + CYL_SIZE + WALL_TOP])
extr = extrapolation.combine_sides(x=extrapolation.BOUNDARY, y=extrapolation.ZERO)
#velocity = StaggeredGrid(math.zeros(batch(batch=3)), extrapolation=BOUNDARY, **DOMAIN)
velocity = StaggeredGrid((VEL,0), extrapolation=extr, **DOMAIN)
#velocity = StaggeredGrid((0,0), extrapolation=extr, **DOMAIN)
pressure = None
BOUNDARY_MASK = StaggeredGrid(HardGeometryMask(Box[:0.2*CYL_SIZE, :]), extrapolation=extr, **DOMAIN)
OBSTACLE = Obstacle(Sphere(center=(WALL_LEFT + 0.5*CYL_SIZE, WALL_BOTTOM + 0.5*CYL_SIZE), radius=0.5*CYL_SIZE))
OBS_MASK = StaggeredGrid(OBSTACLE.geometry, extrapolation=extrapolation.ZERO, **DOMAIN)

RESAMPLING_CENTERED = CenteredGrid(0, extrapolation=extr, **DOMAIN)
RESAMPLING_STAGGERED = StaggeredGrid(math.zeros(channel(vector=2)), extrapolation=extr, **DOMAIN)

VEL_INIT = 0.5 * (np.cos( math.PI * np.arange(0,1,1/RES_Y)[None,...]) + 1)
VEL_INIT = np.repeat(VEL_INIT, RES_X, axis=0)
VEL_INIT = np.stack([VEL_INIT, np.zeros_like(VEL_INIT)], axis=0)
VEL_INIT = CenteredGrid(tensor(VEL_INIT, channel("vector"), spatial("x", "y")),extrapolation=extr, **DOMAIN)
VEL_INIT = StaggeredGrid(VEL_INIT @ RESAMPLING_STAGGERED, extrapolation=extr, **DOMAIN)


### MAIN LOOP
viewer = view("velocity,pressure,VEL_INIT", namespace=globals(), select="batch", gui=gui, keep_alive=(gui is "dash"))
time.sleep(0.01)
print()
print(scene.path)

recVel = []
recPres = []
recVisc = []
recRey = []

for step in viewer.range(STEPS):
    print("\t%s Frame %04d" % ("Reading" if readOnly else "Simulating", step))

    startReyChange = 300
    if REYNOLDS_START == REYNOLDS_END or step < startReyChange:
        visc = VISC_START
        rey = REYNOLDS_START
    else:
        visc = VISC_START + (float(step-startReyChange) / float(STEPS-startReyChange-1)) * (VISC_END - VISC_START)
        rey = REYNOLDS_START + (float(step-startReyChange) / float(STEPS-startReyChange-1)) * (REYNOLDS_END - REYNOLDS_START)
    recVisc += [visc]
    recRey += [rey]

    if not readOnly:
        # simulate
        velocity = advect.mac_cormack(velocity, velocity, DT)
        #velocity = advect.semi_lagrangian(velocity, velocity, DT)
        if step < WARMUP:
            velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * VEL * VEL_INIT
        else:
            velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (VEL, 0)
        velocity, pressure = fluid.make_incompressible(velocity, (OBSTACLE,), Solve("CG-adaptive", 1e-5, 0, max_iterations=2000, x0=pressure))
        velocity = diffuse.explicit(velocity, visc, DT, substeps=int(max(2000*visc,1)))

        # preview image
        velNp = (velocity @ RESAMPLING_CENTERED).values.numpy("vector,x,y")
        maxVelocityMag = np.max(np.sqrt(velNp[0]*velNp[0] + velNp[1]*velNp[1]))
        if maxVelocityMag > 10*VEL:
            print("WARNING: Unusual velocity magnitude detected!")
        if step > 0:
            velNp = np.transpose(velNp, axes=[2,1,0])
            for i in range(velNp.shape[2]):
                velPart = velNp[...,i]
                vMax = max(abs(np.min(velPart)), abs(np.max(velPart)))
                vMin = -vMax
                velPart = 255*((velPart - vMin) / (vMax - vMin))
                imageio.imwrite("data/preview%s.png" % ("X" if i==0 else "Y"), velPart.astype(np.uint8))

        if write:
            # write simulation data
            velNp = (velocity @ RESAMPLING_CENTERED).values.numpy("vector,x,y")
            presNp = pressure.values.numpy("vector,x,y")
            np.savez_compressed( os.path.join(scene.path, "velocity_%06d.npz" % step), velNp.astype(np.float32))
            np.savez_compressed( os.path.join(scene.path, "pressure_%06d.npz" % step), presNp.astype(np.float32))

            # obstacle mask
            if not os.path.isfile(os.path.join(scene.path, "obstacle_mask.npz")):
                obsNp = (OBS_MASK @ RESAMPLING_CENTERED).values.numpy("vector,x,y")
                obsNp = obsNp[0] <= 0
                np.savez_compressed( os.path.join(scene.path, "obstacle_mask.npz"), obsNp.astype(np.int))

    else:
        # read existing simulation data
        velNp = np.load(os.path.join(scene.path, "velocity_%06d.npz" % step))["arr_0"]
        presNp = np.load(os.path.join(scene.path, "pressure_%06d.npz" % step))["arr_0"]
        velGrid = CenteredGrid(tensor(velNp, channel("vector"), spatial("x", "y")), extrapolation=extr, **DOMAIN)
        velocity = StaggeredGrid(velGrid @ RESAMPLING_STAGGERED, extrapolation=extr, **DOMAIN)
        pressure = CenteredGrid(tensor(presNp, channel("vector"), spatial("x", "y")), extrapolation=extr, **DOMAIN)

    recVel += [velNp]
    recPres += [presNp]

recVel = np.transpose(np.stack(recVel, axis=0), axes=[0,2,3,1])
recPres = np.transpose(np.stack(recPres, axis=0), axes=[0,2,3,1])


### RENDERING
if render:
    print("\nRendering...")

    renderpath = os.path.join(scene.path, "render")
    if not os.path.exists(renderpath):
        os.makedirs(renderpath)
    if REYNOLDS_START == REYNOLDS_END:
        renderfile = "cyl%1.2f_vel%1.2f_visc%1.8f_rey%06d" % (CYL_SIZE, VEL, VISC_START, REYNOLDS_START)
    else:
        renderfile = "cyl%1.2f_vel%1.2f_rey%06d-rey%06d" % (CYL_SIZE, VEL, REYNOLDS_START, REYNOLDS_END)

    vx_dx, vx_dy = np.gradient(recVel[...,0][...,None], axis=(1,2))
    vy_dx, vy_dy = np.gradient(recVel[...,1][...,None], axis=(1,2))
    curl = vy_dx - vx_dy
    divergence = vx_dx + vy_dy

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

    vidfile = renderfile + ".mp4"
    imageio.mimwrite(os.path.join(renderpath, vidfile), result, quality=10, fps=int(1/DT), ffmpeg_log_level="error")
    if writeImageSequence:
        for i in range(0,result.shape[0],10):
            imgfile = "img_%06d_%s.png" % (i, renderfile)
            imageio.imwrite(os.path.join(renderpath, imgfile), result[i])


### LOGGING
if not readOnly:
    log = {}

    log["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log["Resolution"] = [RES_X, RES_Y]
    log["Dt"] = DT
    log["Steps, Warmup"] = [STEPS, WARMUP]
    log["Cylinder Size"] = CYL_SIZE
    log["Walls (lrtb)"] = [WALL_LEFT, WALL_RIGHT, WALL_TOP, WALL_BOTTOM]
    log["Inflow Velocity"] = VEL
    log["Fluid Viscosity"] = VISC_START if REYNOLDS_START == REYNOLDS_END else recVisc
    log["Reynolds Number"] = REYNOLDS_START if REYNOLDS_START == REYNOLDS_END else recRey
    log["Stats"] = {"Velocity" : [], "Velocity Magnitude" : [], "Pressure" : []}

    recVelMag = np.linalg.norm(recVel, axis=-1)
    for i in range(recVel.shape[0]):
        log["Stats"]["Velocity"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recVel[i]), np.max(recVel[i]), np.mean(recVel[i])) )
        log["Stats"]["Velocity Magnitude"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recVelMag[i]), np.max(recVelMag[i]), np.mean(recVelMag[i])) )
        log["Stats"]["Pressure"].append( "Min:%2.8f Max:%2.8f Avg: %2.8f" % (np.min(recPres[i]), np.max(recPres[i]), np.mean(recPres[i])) )

    if not readOnly:
        logFile = os.path.join(scene.path, "src", "description.json")
        with open(logFile, 'w') as f:
            json.dump(log, f, indent=4)
            f.close()


print("Simulation complete\n\n")

if gui == "console":
    os._exit(0)