# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:35:19 2019

@author: Bruna Campos

Sepearmen Correlation x Pearson Correlation
"""

import numpy as np
import pandas as pd
import scipy.stats as sp
import glob
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.stats import pearsonr


asd = glob.glob("ASD*.txt")
td = glob.glob("TD*.txt")
AT = np.zeros((754,1))
AA = np.zeros((841,1))
TT = np.zeros((676,1))
i = 0
j = 0
k = 0

for A in asd:
    
    for T in td:
    
        df = pd.read_csv(A, sep="\t", header=None)
        #df1 = np.transpose(df)
        
        dff = pd.read_csv(T, sep="\t", header=None)
        #df2 = np.transpose(dff)
        
        dfa = np.corrcoef(df)
        dfb = np.corrcoef(dff)
        
        
        a = np.reshape(dfa, 69696, 'C')
        b = np.reshape(dfb, 69696, 'C')
        
        
        rho, pvalue = spearmanr(a, b)
        r, rr = pearsonr(a, b)
        
        AT[i] = rho
        i = i + 1
        
for A in asd:
    
    for S in asd:
    
        df = pd.read_csv(A, sep="\t", header=None)
        #df1 = np.transpose(df)
        
        dff = pd.read_csv(S, sep="\t", header=None)
        #df2 = np.transpose(dff)
        
        dfa = np.corrcoef(df)
        dfb = np.corrcoef(dff)
        
        
        a = np.reshape(dfa, 69696, 'C')
        b = np.reshape(dfb, 69696, 'C')
        
        
        rho, pvalue = spearmanr(a, b)
        r, rr = pearsonr(a, b)
        
        AA[j] = rho
        j = j + 1
        
for D in td:
    
    for T in td:
    
        df = pd.read_csv(D, sep="\t", header=None)
        #df1 = np.transpose(df)
        
        dff = pd.read_csv(T, sep="\t", header=None)
        #df2 = np.transpose(dff)
        
        dfa = np.corrcoef(df)
        dfb = np.corrcoef(dff)
        
        
        a = np.reshape(dfa, 69696, 'C')
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
plt.title("Spearman Correlation in Pearson Correlation")
plt.grid(True)
plt.show()
plt.savefig('spearman_pearson.png')


#Medium ASDxASD 0.35152747052753824 +- 0.1441408919888256
#Medium TDxTD 0.4205323517086612 +- 0.1306437181181694
#Medium TDxASD 0.3578639448783557 +- 0.07196015182635074
