# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:22:16 2019

@author: Bruna Campos

Sepearmen Correlation x Partial Correlation
"""
import numpy as np
import pandas as pd
import scipy as sp
import glob
import matplotlib.pyplot as plt
import pingouin as pg

from scipy.stats import spearmanr, pearsonr


asd_dataset = glob.glob("ASD*.txt")
td_dataset = glob.glob("TD*.txt")
AT = np.zeros((754,1))
AA = np.zeros((841,1))
TT = np.zeros((676,1))
i = 0
j = 0
k = 0

for asd_data in asd_dataset:
    
    df = pd.read_csv(asd_data, sep="\t", header=None).T
    
    partial1 = df.pcorr()
    
    dfa = np.asarray(partial1)
    a = np.reshape(dfa, 69696, 'C')
    
    for td_data in td_dataset:
        
        dff = pd.read_csv(td_data, sep="\t", header=None).T
        
        partial2 = dff.pcorr()
        
        dfb = np.asarray(partial2)
        
        b = np.reshape(dfb, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        r, rr = pearsonr(a, b)
        
        AT[i] = rho
        i = i + 1
        
for asd_data in asd_dataset:
    
    df = pd.read_csv(asd_data, sep="\t", header=None).T
    
    partial1 = df.pcorr()
    
    dfa = np.asarray(partial1)
    a = np.reshape(dfa, 69696, 'C')
    
    for asd_data in asd_dataset:
        
        dff = pd.read_csv(asd_data, sep="\t", header=None).T
        
        partial2 = dff.pcorr()
        
        dfb = np.asarray(partial2)
        
        b = np.reshape(dfb, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        r, rr = pearsonr(a, b)
        
        AA[j] = rho
        j = j + 1
        
for td_data in td_dataset:
    
    df = pd.read_csv(td_data, sep="\t", header=None).T
    
    partial1 = df.pcorr()
    
    dfa = np.asarray(partial1)
    a = np.reshape(dfa, 69696, 'C')
    
    for td_data in td_dataset:
        
        dff = pd.read_csv(td_data, sep="\t", header=None).T
        
        partial2 = dff.pcorr()
        
        dfb = np.asarray(partial2)
        
        b = np.reshape(dfb, 69696, 'C')
        
        rho, pvalue = spearmanr(a, b)
        r, rr = pearsonr(a, b)
        
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
plt.title("Spearman Correlation in Partial Correlation")
plt.grid(True)
plt.show()
plt.savefig('spearman_partial.png')


#Medium ASDxASD 0.05027569565995945 +- 0.17956945056872986
#Medium TDxTD 0.05435268027726336 +- 0.18922854702968778
#Medium TDxASD 0.016576864055881824 +- 0.00587412132480237