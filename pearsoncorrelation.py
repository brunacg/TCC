# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:10:54 2019

@author: Bruna Campos

Pearson's r correlation
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import glob

# import scipy as sp

from matplotlib import cm as cm
from scipy import stats

#%%Functions

def pearson_corr(df):

    fmatrix = df.values
    rows, cols = fmatrix.shape

    r = np.ones((cols, cols), dtype=float)
    p = np.ones((cols, cols), dtype=float)

    for i in range(cols):
        for j in range(cols):
            if i == j:
                r_, p_ = 1., 1.
            else:
                r_, p_ = sp.stats.pearsonr(fmatrix[:,i], fmatrix[:,j])

            r[j][i] = r_
            p[j][i] = p_

    return r

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
    plt.savefig('pearsoncorr'+name+'.png')

#%%fechando

asd_dataset = glob.glob("ASD*.txt")
td_dataset = glob.glob("TD*.txt")
figs = 0
figa = 0

for asd_data in asd_dataset:

    # Read file into a Pandas dataframe
    df1 = pd.read_csv(asd_data, sep="\t", header=None).T
    #df = np.transpose(df1)
    
    r1 = pearson_corr(df1)
    
    names = figs
    figs = figs + 1
    
    correlation_matrix(r1, 'Pearson Correlation ' + asd_data, 'asd' + str(names))

for td_data in td_dataset:

    df2 = pd.read_csv(td_data, sep="\t", header=None).T
    #dfs = np.transpose(df2)

    r2 = pearson_corr(df2)
    
    name = figa
    figa = figa + 1

    correlation_matrix(r2, 'Pearson Correlation ' + td_data, 'td' + str(name))

