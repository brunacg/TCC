#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:28:51 2019

@author: bruna

Mutual Information
"""
import numpy as np
import pandas as pd
import scipy as sp
import glob
import matplotlib.pyplot as plt

from matplotlib import cm as cm

#%%Functions

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = (H_X + H_Y - H_XY)
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def correlation_matrix(df, title, name):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap("Oranges")
    cax = ax1.imshow(
        df,
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=0.6,
        origin="lower",
    )
    # ax1.grid(True)
    plt.title(title)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ax=ax1)
    plt.show()
    plt.savefig('mutualinfo'+name+'.png')
    
#%%

asd_dataset = glob.glob("ASD*.txt")
td_dataset = glob.glob("TD*.txt")
figs = 0
figa = 0
bins = 8

for asd_data in asd_dataset:

    # Read file into a Pandas dataframe
    df1 = np.asarray(pd.read_csv(asd_data, sep="\t", header=None).T)
    n = df1.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(df1[:,ix], df1[:,jx], bins)
            matMI[jx,ix] = calc_MI(df1[:,ix], df1[:,jx], bins)
    
    names = figs
    figs = figs + 1
    
    correlation_matrix(matMI, 'Mutual Information ' + asd_data, 'asd' + str(names))

for td_data in td_dataset:

    df2 = np.asarray(pd.read_csv(td_data, sep="\t", header=None).T)

    n = df2.shape[1]
    matMI2 = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI2[ix,jx] = calc_MI(df2[:,ix], df2[:,jx], bins)
            matMI2[jx,ix] = calc_MI(df2[:,ix], df2[:,jx], bins)
    
    name = figa
    figa = figa + 1

    correlation_matrix(matMI2, 'Mutual Information ' + td_data, 'td' + str(name))
