# -*- coding: utf-8 -*-
"""
Created on Fri May 17 05:23:26 2019

@author: Bruna Campos

Partial Correlation
"""
import numpy as np
import pandas as pd
#import time 
import matplotlib.pyplot as plt
import glob
import pingouin as pg

from matplotlib import cm as cm

#%%Functions

def correlation_matrix(df, title, name):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap("seismic")
    cax = ax1.imshow(
        df,
        interpolation="nearest",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    # ax1.grid(True)
    plt.title(title)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ax=ax1)
    plt.show()
    plt.savefig('partialcorr'+name+'.png')

#%%


asd_dataset = glob.glob("ASD*.txt")
td_dataset = glob.glob("TD*.txt")
figs = 0
figa = 0

for asd_data in asd_dataset:

    df1 = pd.read_csv(asd_data, sep="\t", header=None).T
    
    p_corr1 = df1.pcorr()
    
    names = figs
    figs = figs + 1
    
    correlation_matrix(p_corr1, 'Partial Correlation ' + asd_data, 'asd' + str(names))

for td_data in td_dataset:

    df2 = pd.read_csv(td_data, sep="\t", header=None).T

    p_corr2 = df2.pcorr()
    
    name = figa
    figa = figa + 1

    correlation_matrix(p_corr2, 'Partial Correlation ' + td_data, 'td' + str(name))
