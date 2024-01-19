

class DataParams(object):
    def __init__(self, batch=4, augmentations=[], sequenceLength=[1,1], randSeqOffset=False,
                dataSize=[128,64], dimension=2, simFields=[], simParams=[], normalizeMode=""):
        self.batch          = batch             # batch size
        self.augmentations  = augmentations     # used data augmentations
        self.sequenceLength = sequenceLength    # number of simulation frames in one sequence
        self.randSeqOffset  = randSeqOffset     # randomize sequence starting frame
        self.dataSize       = dataSize          # target data size for scale/crop/cropRandom transformation
        self.dimension      = dimension         # number of data dimension
        self.simFields      = simFields         # which simulation fields are added (vel is always used) from ["dens", "pres"]
        self.simParams      = simParams         # which simulation parameters are added from ["rey", "mach"]
        self.normalizeMode  = normalizeMode     # which mean and std values from different data sets are used in normalization transformation

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.batch          = d.get("batch",            -1)
        p.augmentations  = d.get("augmentations",    [])
        p.sequenceLength = d.get("sequenceLength",   [])
        p.randSeqOffset  = d.get("randSeqOffset",    False)
        p.dataSize       = d.get("dataSize",         -1)
        p.dimension      = d.get("dimension",        -1)
        p.simFields      = d.get("simFields",        [])
        p.simParams      = d.get("simParams",        [])
        p.normalizeMode  = d.get("normalizeMode",    "")
        return p

    def asDict(self) -> dict:
        return {
            "batch"          : self.batch,
            "augmentations"  : self.augmentations,
            "sequenceLength" : self.sequenceLength,
            "randSeqOffset"  : self.randSeqOffset,
            "dataSize"       : self.dataSize,
            "dimension"      : self.dimension,
            "simFields"      : self.simFields,
            "simParams"      : self.simParams,
            "normalizeMode"  : self.normalizeMode,
        }



class TrainingParams(object):
    def __init__(self, epochs=20, lr=0.0001, expLrGamma=1.0, weightDecay=0.0, fadeInPredLoss=[-1,0], fadeInSeqLen=[-1,0], fadeInSeqLenLin=False):
        self.epochs            = epochs            # number of training epochs
        self.lr                = lr                # learning rate
        self.expLrGamma        = expLrGamma        # factor for exponential learning rate decay
        self.weightDecay       = weightDecay       # weight decay factor to regularize the net by penalizing large weights
        self.fadeInPredLoss    = fadeInPredLoss    # start and end epoch of fading in the prediction loss terms
        self.fadeInSeqLen      = fadeInSeqLen      # start and end epoch of fading in the sequence length
        self.fadeInSeqLenLin   = fadeInSeqLenLin   # exponential or linear scaling of fading in the sequence length
        
    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.epochs            = d.get("epochs",           -1)
        p.lr                = d.get("lr",               -1)
        p.expLrGamma        = d.get("expLrGamma",        1)
        p.weightDecay       = d.get("weightDecay",      -1)
        p.fadeInPredLoss    = d.get("fadeInPredLoss",   [])
        p.fadeInSeqLen      = d.get("fadeInSeqLen",     [])
        p.fadeInSeqLenLin   = d.get("fadeInSeqLenLin",  False)
        return p

    def asDict(self) -> dict:
        return {
            "epochs"            : self.epochs,
            "lr"                : self.lr,
            "expLrGamma"        : self.expLrGamma,
            "weightDecay"       : self.weightDecay,
            "fadeInPredLoss"    : self.fadeInPredLoss,
            "fadeInSeqLen"      : self.fadeInSeqLen,
            "fadeInSeqLenLin"   : self.fadeInSeqLenLin,
        }



class LossParams(object):
    def __init__(self, recMSE=1.0, recLSIM=0, predMSE=1.0, predLSIM=0, extraMSEvelZ=0, regMeanStd=0, regDiv=0, regVae=0, regLatStep=0):
        self.recMSE       = recMSE       # mse loss reconstruction weight
        self.recLSIM      = recLSIM      # lsim loss reconstruction weight
        self.predMSE      = predMSE      # mse loss prediction weight
        self.predLSIM     = predLSIM     # lsim loss prediction weight
        self.regMeanStd   = regMeanStd   # mean and standard deviation regularization weight
        self.regDiv       = regDiv       # divergence regularization weight
        self.regVae       = regVae       # regularization weight for VAE KL divergence
        self.regLatStep   = regLatStep   # latent space step regularization weight

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.recMSE       = d.get("recMSE", -1)
        p.recLSIM      = d.get("recLSIM", -1)
        p.predMSE      = d.get("predMSE", -1)
        p.predLSIM     = d.get("predLSIM", -1)
        p.regMeanStd   = d.get("regMeanStd", -1)
        p.regDiv       = d.get("regDiv", -1)
        p.regVae       = d.get("regVae", -1)
        p.regLatStep   = d.get("regLatStep", -1)
        return p

    def asDict(self) -> dict:
        return {
            "recMSE"       : self.recMSE,
            "recLSIM"      : self.recLSIM,
            "predMSE"      : self.predMSE,
            "predLSIM"     : self.predLSIM,
            "regMeanStd"   : self.regMeanStd,
            "regDiv"       : self.regDiv,
            "regVae"       : self.regVae,
            "regLatStep"   : self.regLatStep,
        }



class ModelParamsEncoder(object):
    def __init__(self, arch="skip", pretrained=False, frozen=False, encWidth=16, latentSize=16):
        self.arch = arch              # architecture variant
        self.pretrained = pretrained  # load pretrained weight initialization
        self.frozen = frozen          # freeze weights after initialization
        self.encWidth = encWidth      # width of encoder network
        self.latentSize = latentSize  # size of latent space vector

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch       = d.get("arch", "")
        p.pretrained = d.get("pretrained", False)
        p.frozen     = d.get("frozen", False)
        p.encWidth   = d.get("encWidth", -1)
        p.latentSize = d.get("latentSize", -1)
        return p

    def asDict(self) -> dict:
        return {
            "arch"       : self.arch,
            "pretrained" : self.pretrained,
            "frozen"     : self.frozen,
            "encWidth"   : self.encWidth,
            "latentSize" : self.latentSize,
        }



class ModelParamsDecoder(object):
    def __init__(self, arch="skip", pretrained=False, frozen=False, decWidth=48, vae=False, trainingNoise=0.0,
                 diffSteps=500, diffSchedule="linear", diffCondIntegration="noisy", fnoModes=(16,16), refinerStd=0.0):
        self.arch = arch                 # architecture variant
        self.pretrained = pretrained     # load pretrained weight initialization
        self.frozen = frozen             # freeze weights after initialization
        self.decWidth = decWidth         # width of decoder network
        self.vae = vae                   # use a variational AE setup
        self.trainingNoise = trainingNoise # amount of training noise added to inputs
        self.diffSteps = diffSteps       # diffusion model diffusion time steps
        self.diffSchedule = diffSchedule # diffusion model variance schedule
        self.diffCondIntegration = diffCondIntegration # integrationg of conditioning during diffusion training
        self.fnoModes = fnoModes         # number of fourier modes for FNO setup
        self.refinerStd = refinerStd     # noise standard dev. in pde refiner setup

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch         = d.get("arch", "")
        p.pretrained   = d.get("pretrained", False)
        p.frozen       = d.get("frozen", False)
        p.decWidth     = d.get("decWidth", -1)
        p.vae          = d.get("vae", False)
        p.trainingNoise= d.get("trainingNoise", 0.0)
        p.diffSteps    = d.get("diffSteps", 500)
        p.diffSchedule = d.get("diffSchedule", "linear")
        p.diffCondIntegration  = d.get("diffCondIntegration", "noisy")
        p.fnoModes     = d.get("fnoModes", ())
        p.refinerStd   = d.get("refinerStd", 0.0)
        return p

    def asDict(self) -> dict:
        return {
            "arch"         : self.arch,
            "pretrained"   : self.pretrained,
            "frozen"       : self.frozen,
            "decWidth"     : self.decWidth,
            "vae"          : self.vae,
            "trainingNoise": self.trainingNoise,
            "diffSteps"    : self.diffSteps,
            "diffSchedule" : self.diffSchedule,
            "diffCondIntegration" : self.diffCondIntegration,
            "fnoModes"     : self.fnoModes,
            "refinerStd"   : self.refinerStd,
        }



class ModelParamsLatent(object):
    def __init__(self, arch="fc", pretrained=False, frozen=False, width=512, layers=6, heads=4, dropout=0.0,
               transTrainUnroll=False, transTargetFull=False, maxInputLen=-1):
        self.arch = arch                         # architecture variant
        self.pretrained = pretrained             # load pretrained weight initialization
        self.frozen = frozen                     # freeze weights after initialization
        self.width = width                       # latent network width
        self.layers = layers                     # number of latent network layers
        self.heads = heads                       # number of attention heads in transformer
        self.dropout = dropout                   # dropout rate in latent network
        self.transTrainUnroll = transTrainUnroll # unrolled training for transformer latent models, FALSE for one step predictions TRUE for full rollouts
        self.transTargetFull = transTargetFull   # full target data for transformer and transformer decoder latent models, FALSE for only the previous step as a target TRUE for every previous step as a target
        self.maxInputLen = maxInputLen           # how many steps of the input sequence are processed at once for models that predict full sequences (-1 for no limit)


    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.arch             = d.get("arch", "")
        p.pretrained       = d.get("pretrained", False)
        p.frozen           = d.get("frozen", False)
        p.width            = d.get("width", "")
        p.layers           = d.get("layers", "")
        p.heads            = d.get("heads", "")
        p.dropout          = d.get("dropout", "")
        p.transTrainUnroll = d.get("transTrainUnroll", False)
        p.transTargetFull  = d.get("transTargetFull", False)
        p.maxInputLen      = d.get("maxInputLen", -1)
        return p

    def asDict(self) -> dict:
        return {
            "arch"             : self.arch,
            "pretrained"       : self.pretrained,
            "frozen"           : self.frozen,
            "width"            : self.width,
            "layers"           : self.layers,
            "heads"            : self.heads,
            "dropout"          : self.dropout,
            "transTrainUnroll" : self.transTrainUnroll,
            "transTargetFull"  : self.transTargetFull,
            "maxInputLen"      : self.maxInputLen,
        }