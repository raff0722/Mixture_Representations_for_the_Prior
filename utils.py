import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

def load(file):
    file = Path(file)
    file = open(file, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable

def save(file, variable):
    file = Path(file)
    file = open(file, 'wb')
    pickle.dump(variable, file)
    file.close()

fast_exponential = lambda rate, N: - np.log( np.random.random( size=(rate.size, N) ) ) / rate[:, None]
fast_Laplace = lambda rate, N: fast_exponential(rate, N) - fast_exponential(rate, N)

def inv_prob(grid=1, fig_width=7.5, fig_height=5):

    cm = 1/2.54
    plt.rcdefaults()

    rcParams = {\
    'lines.linewidth':.7,\
    'lines.markersize':1.9,\
    'legend.fontsize':9,\
    'legend.fancybox':False,\
    'legend.framealpha':1,\
    'xtick.labelsize':7,\
    'ytick.labelsize':7,\
    'text.usetex':False,\
    'font.family':'serif',\
    # 'font.name': 'Computer Modern Roman',\
    'mathtext.fontset':'cm',\
    # 'mathtext':'regular',\
    'figure.figsize':(fig_width*cm, fig_height*cm),\
    'font.size':9,\
    'axes.prop_cycle': plt.cycler(color=plt.cm.tab10.colors)
    }

    if grid:
        rcParams.update({
            'axes.grid':True,
            'grid.linewidth':.3,
            'grid.color':'lightgrey'
            })

    plt.rcParams.update(rcParams)