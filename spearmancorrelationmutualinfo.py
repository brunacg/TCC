#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:33:09 2019

@author: bruna

Sepearman Correlation Mutual Information
"""
import numpy as np
import pandas as pd
import scipy as sp
import glob
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

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

asd_dataset = glob.glob("ASD*.txt")
td_dataset = glob.glob("TD*.txt")
AT = np.zeros((754,1))
AA = np.zeros((841,1))
TT = np.zeros((676,1))
i = 0
j = 0
k = 0
bins = 8

for asd_data in asd_dataset:
    
    df1 = np.asarray(pd.read_csv(asd_data, sep="\t", header=None).T)
    
    n = df1.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(df1[:,ix], df1[:,jx], bins)
            matMI[jx,ix] = calc_MI(df1[:,ix], df1[:,jx], bins)
    
    a = np.reshape(matMI, 69696, 'C')
    
    for td_data in td_dataset:
        
        df2 = np.asarray(pd.read_csv(td_data, sep="\t", header=None).T)
    
        n = df2.shape[1]
        matMI2 = np.zeros((n, n))

        for ix in np.arange(n):
            for jx in np.arange(ix+1,n):
                matMI2[ix,jx] = calc_MI(df2[:,ix], df2[:,jx], bins)
                matMI2[jx,ix] = calc_MI(df2[:,ix], df2[:,jx], bins)
    
        b = np.reshape(matMI2, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        
        AT[i] = rho
        i = i + 1
        
for asd_data in asd_dataset:
    
    df1 = np.asarray(pd.read_csv(asd_data, sep="\t", header=None).T)
    
    n = df1.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(df1[:,ix], df1[:,jx], bins)
            matMI[jx,ix] = calc_MI(df1[:,ix], df1[:,jx], bins)
    
    a = np.reshape(matMI, 69696, 'C')
    
    for asd_data in asd_dataset:
        
        df2 = np.asarray(pd.read_csv(asd_data, sep="\t", header=None).T)
    
        n = df2.shape[1]
        matMI2 = np.zeros((n, n))

        for ix in np.arange(n):
            for jx in np.arange(ix+1,n):
                matMI2[ix,jx] = calc_MI(df2[:,ix], df2[:,jx], bins)
                matMI2[jx,ix] = calc_MI(df2[:,ix], df2[:,jx], bins)
    
        b = np.reshape(matMI2, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        
        AA[j] = rho
        j = j + 1
        
for td_data in td_dataset:
    
    df1 = np.asarray(pd.read_csv(td_data, sep="\t", header=None).T)
    
    n = df1.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(df1[:,ix], df1[:,jx], bins)
            matMI[jx,ix] = calc_MI(df1[:,ix], df1[:,jx], bins)
    
    a = np.reshape(matMI, 69696, 'C')
    
    for td_data in td_dataset:
        
        df2 = np.asarray(pd.read_csv(td_data, sep="\t", header=None).T)
    
        n = df2.shape[1]
        matMI2 = np.zeros((n, n))

        for ix in np.arange(n):
            for jx in np.arange(ix+1,n):
                matMI2[ix,jx] = calc_MI(df2[:,ix], df2[:,jx], bins)
                matMI2[jx,ix] = calc_MI(df2[:,ix], df2[:,jx], bins)
    
        b = np.reshape(matMI2, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        
        TT[k] = rho
        k = k + 1
    
media_AT = AT.mean()
media_AA = AA.mean()
media_TT = TT.mean()

dp_AT = AT.std()
dp_AA = AA.std()
dp_TT = TT.std()

x = ("ASDxASD", "TDxTD", "TDxASD")
y = np.array([media_AA, media_TT, media_AT])
e = np.array([dp_AA, dp_TT, dp_AT])

plt.errorbar(x, y, yerr = e, linestyle='None', fmt='o')
plt.yscale('linear')
plt.title("Spearman Correlation in Mutual Information")
plt.grid(True)
plt.show()
plt.savefig('spearman_mutualinfo.png')

#Medium TDxASD 0.18380046712510303 +- 0.0679523439616953
#Medium ASDxASD 0.19764824848551593 +- 0.16634452377404546
#Medium TDxTD 0.2348846923954082 +- 0.1644752455189415