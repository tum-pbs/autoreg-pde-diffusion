import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

colorRemap = {
"Simulation": "k",
"Sim.": "k",
"Simulation-full": "k",

"TF-MGN": "tab:pink",
"TF-Enc": "tab:red",
"TF-VAE": "tab:purple",

"ResNet": "tab:gray",
"Dil-ResNet": "tab:cyan",

"FNO16": "tab:green",
"FNO32": "tab:olive",

"U-Net": "tab:blue",
"U-Net-m4":      mpl.colormaps["turbo"].resampled(10)(range(10))[3],
"U-Net-m8":      mpl.colormaps["turbo"].resampled(10)(range(10))[5],
"U-Net-m16":     mpl.colormaps["turbo"].resampled(10)(range(10))[7],
"U-Net-m4-Pre":  mpl.colormaps["turbo"].resampled(14)(range(14))[5],
"U-Net-m8-Pre":  mpl.colormaps["turbo"].resampled(14)(range(14))[7],
"U-Net-m16-Pre": mpl.colormaps["turbo"].resampled(14)(range(14))[9],

"U-Net-n0.0001": mpl.colormaps["viridis"].resampled(25)(range(25))[13],
"U-Net-n0.001":  mpl.colormaps["viridis"].resampled(25)(range(25))[16],
"U-Net-n0.01":   mpl.colormaps["viridis"].resampled(25)(range(25))[20],
"U-Net-n0.1":    mpl.colormaps["viridis"].resampled(25)(range(25))[22],
"U-Net-n1.0":    mpl.colormaps["viridis"].resampled(25)(range(25))[24],

"ACDM": "tab:orange",
"ACDM-R500": mpl.colormaps["plasma"].resampled(12)(range(12))[0],
"ACDM-R100": mpl.colormaps["plasma"].resampled(12)(range(12))[2],
"ACDM-R50":  mpl.colormaps["plasma"].resampled(12)(range(12))[4],
"ACDM-R30":  mpl.colormaps["plasma"].resampled(12)(range(12))[6],
"ACDM-R20":  mpl.colormaps["plasma"].resampled(12)(range(12))[8],
"ACDM-R15":  mpl.colormaps["plasma"].resampled(12)(range(12))[9],
"ACDM-R10":  mpl.colormaps["plasma"].resampled(12)(range(12))[10],

"ACDM-ncn": "tab:brown",
"ACDM-ncn-n0.0001": mpl.colormaps["terrain_r"].resampled(20)(range(20))[7],
"ACDM-ncn-n0.001":  mpl.colormaps["terrain_r"].resampled(20)(range(20))[10],
"ACDM-ncn-n0.01":   mpl.colormaps["terrain_r"].resampled(20)(range(20))[13],
"ACDM-ncn-n0.1":    mpl.colormaps["terrain_r"].resampled(20)(range(20))[16],
"ACDM-ncn-n1.0":    mpl.colormaps["terrain_r"].resampled(20)(range(20))[19],
}

modelRemap = {
"Simulation": r"$\bf{Simulation}$",
"Sim.": r"$\bf{Sim.}$",

"TF-MGN": r"$\mathit{TF}_{\mathit{MGN}}$",
"TF-Enc": r"$\mathit{TF}_{\mathit{Enc}}$",
"TF-VAE": r"$\mathit{TF}_{\mathit{VAE}}$",

"ResNet": r"$\mathit{ResNet}$",
"Dil-ResNet": r"$\mathit{ResNet}_{\mathit{dil.}}$",

"FNO16": r"$\mathit{FNO}_{16}$",
"FNO32": r"$\mathit{FNO}_{32}$",

"U-Net": r"$\mathit{U}$-$\mathit{Net}$",
"U-Net-m4": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m4}}$",
"U-Net-m8": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m8}}$",
"U-Net-m16": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m16}}$",
"U-Net-m4-Pre": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m4,Pre}}$",
"U-Net-m8-Pre": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m8,Pre}}$",
"U-Net-m16-Pre": r"$\mathit{U}$-$\mathit{Net}_{\mathit{m16,Pre}}$",

"ACDM": r"$\mathit{ACDM}$",
"ACDM-bold": r"$\mathbf{ACDM}$",
"ACDM-ncn": r"$\mathit{ACDM}_{\mathit{ncn}}$",
"ACDM-R10": r"$\mathit{ACDM}_{\mathit{R10}}$",
"ACDM-R15": r"$\mathit{ACDM}_{\mathit{R15}}$",
"ACDM-R20": r"$\mathit{ACDM}_{\mathit{R20}}$",
"ACDM-R30": r"$\mathit{ACDM}_{\mathit{R30}}$",
"ACDM-R50": r"$\mathit{ACDM}_{\mathit{R50}}$",
"ACDM-R100": r"$\mathit{ACDM}_{\mathit{R100}}$",
"ACDM-R500": r"$\mathit{ACDM}_{\mathit{R500}}$",

"U-Net-n0.0001": r"$\mathit{U}$-$\mathit{Net}_{\mathit{n1e-4}}$",
"U-Net-n0.001": r"$\mathit{U}$-$\mathit{Net}_{\mathit{n1e-3}}$",
"U-Net-n0.01": r"$\mathit{U}$-$\mathit{Net}_{\mathit{n1e-2}}$",
"U-Net-n0.1": r"$\mathit{U}$-$\mathit{Net}_{\mathit{n1e-1}}$",
"U-Net-n1.0": r"$\mathit{U}$-$\mathit{Net}_{\mathit{n1e0}}$",

"ACDM-ncn-n0.0001": r"$\mathit{ACDM}_{\mathit{ncn,n1e-4}}$",
"ACDM-ncn-n0.001": r"$\mathit{ACDM}_{\mathit{ncn,n1e-3}}$",
"ACDM-ncn-n0.01": r"$\mathit{ACDM}_{\mathit{ncn,n1e-2}}$",
"ACDM-ncn-n0.1": r"$\mathit{ACDM}_{\mathit{ncn,n1e-1}}$",
"ACDM-ncn-n1.0": r"$\mathit{ACDM}_{\mathit{ncn,n1e0}}$",
}

datasetRemap = {
"lowRey"   : r"$\mathtt{Inc}_{\mathtt{low}}$",
"highRey"  : r"$\mathtt{Inc}_{\mathtt{high}}$",
"varReyIn" : r"$\mathtt{Inc}_{\mathtt{var}}$",

"extrap" : r"$\mathtt{Tra}_{\mathtt{ext}}$",
"interp" : r"$\mathtt{Tra}_{\mathtt{int}}$",
"longer" : r"$\mathtt{Tra}_{\mathtt{long}}$",

"zInterp" : r"$\mathtt{Iso}$",
}

fieldIndexRemap = {
"lowRey"   : {"velX": 0, "velY": 1, "pres": 2, "vort": [0,1]},
"highRey"  : {"velX": 0, "velY": 1, "pres": 2, "vort": [0,1]},
"varReyIn" : {"velX": 0, "velY": 1, "pres": 2, "vort": [0,1]},

"extrap" : {"velX": 0, "velY": 1, "dens": 2, "pres": 3, "vort": [0,1]},
"interp" : {"velX": 0, "velY": 1, "dens": 2, "pres": 3, "vort": [0,1]},
"longer" : {"velX": 0, "velY": 1, "dens": 2, "pres": 3, "vort": [0,1]},

"zInterp" : {"velX": 0, "velY": 1, "velZ": 2, "pres": 3, "vort": [0,1]},
}

lossRelevantFieldRemap = {
"lowRey"   : (0,3),
"highRey"  : (0,3),
"varReyIn" : (0,3),

"extrap" : (0,4),
"interp" : (0,4),
"longer" : (0,4),

"zInterp" : (0,4),
}

clampRemap = {
"lowRey"   : {"velX": (-0.3,0.9), "velY": (-0.5,0.5), "pres": (-0.012,0.012), "vort": (-0.25,0.25)},
"highRey"  : {"velX": (-0.3,0.9), "velY": (-0.5,0.5), "pres": (-0.012,0.012), "vort": (-0.25,0.25)},
"varReyIn" : {"velX": (-0.3,0.9), "velY": (-0.5,0.5), "pres": (-0.012,0.012), "vort": (-0.25,0.25)},

"extrap" : {"velX": (-0.4,1.2), "velY": (-0.7,0.7), "dens": (0.3,1.2), "pres": (0.15,0.75), "vort": (-0.3,0.3)},
"interp" : {"velX": (-0.4,1.2), "velY": (-0.7,0.7), "dens": (0.3,1.2), "pres": (0.15,0.75), "vort": (-0.3,0.3)},
"longer" : {"velX": (-0.4,1.2), "velY": (-0.7,0.7), "dens": (0.3,1.2), "pres": (0.15,0.75), "vort": (-0.3,0.3)},

"zInterp" : {"velX": (-1.5,1.5), "velY": (-1.5,1.5), "velZ": (-1.2,0.6), "pres": (-0.7,0.4), "vort": (-1.0,1.0)},
}



def getColor(id:str):
    if not id in colorRemap:
        #colList = list(colors.cnames.keys())
        #colName = random.choice(colList)
        #col = colors.cnames[colName]
        val = random.random()
        col = plt.cm.get_cmap("jet")(val)
        print("WARNING: no plotting color found for model with ID: %s, using color %s instead" % (id, col))
        return col
    else:
        return colorRemap[id]


def getModelName(id:str):
    if not id in modelRemap:
        print("WARNING: no plotting name found for model with ID: %s" % (id))
        return id
    else:
        return modelRemap[id]


def getDatasetName(id:str):
    if not id in datasetRemap:
        print("WARNING: no plotting name found for dataset with ID: %s" % (id))
        return id
    else:
        return datasetRemap[id]


def getFieldIndex(dataset:str, field:str):
    return fieldIndexRemap[dataset][field]


def getLossRelevantFields(dataset:str):
    return lossRelevantFieldRemap[dataset]


def getColormapAndNorm(dataset:str, field:str):
    norm = None

    if field == "vort":
        if dataset in ["lowRey", "highRey", "varReyIn"]:
            cmap = "icefire"

        elif dataset in ["interp", "extrap", "longer"]:
            cmap = "seismic"

        elif dataset in ["zInterp"]:
            top = mpl.colormaps["bone_r"].resampled(128)
            bottom = mpl.colormaps["hot"].resampled(128)
            newcolors = np.vstack((top(np.linspace(0.1, 1, 128)), bottom(np.linspace(0, 0.9, 128))))
            cmap = ListedColormap(newcolors, name="custom")

            vMin, vMax = clampRemap[dataset][field]
            norm = mpl.colors.SymLogNorm(linthresh=0.05, base=2,vmin=vMin, vmax=vMax)

        else:
            raise ValueError("Unknown dataset: %s" % dataset)

    elif field == "dens":
        cmap = "Spectral"

    elif field == "pres":
        cmap = "Spectral_r"

    elif field in ["velX", "velY", "velZ"]:
        cmap = "viridis"

    else:
        raise ValueError("Unknown field: %s" % field)

    if norm is not None:
        return (cmap, norm)
    else:
        vMin, vMax = clampRemap[dataset][field]
        defaultNorm = mpl.colors.Normalize(vmin=vMin, vmax=vMax)
        return (cmap, defaultNorm)