#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.utils.extmath import randomized_svd
from pysmooth import smooth        
import numpy as np



def match_arg(x, lst):
    return  [el for el in lst if x in el]
    

def rowNorm(x):
    s = np.std(x, axis = 1)
    m = x.mean(axis = 1)
    x = (x.T-m)/s
    return x.T
    
    

def num_pc(data, method = 'elbow', B = 20, seed = None):
    method = match_arg(method, ['elbow', 'permutation'])
    if seed is None:
        np.random.seed(seed)

    #computing SVD
    m, n = data.shape
    data = rowNorm(data)
    if n < 500:
        k = n
    else:
        k = max(200, n/4)
    
    uu = randomized_svd(data, n_components = k, random_state = 123456, n_iter = 3)
    #7129*10, 10, 10*72
    #############################################
    if method == 'permutation':
        dstat = uu[1]**2/np.sum(uu[1]**2)
        dstat0 = np.zeros([B,k])
        for i in range(B):
            dat0 = [np.random.choice(data[el,],size = n, replace = False) for el in range(m)]
            dat0 = np.array(dat0)
            uu0 = randomized_svd(dat0, n_components = k, random_stat = 123456, n_iter = 3)
            dstat0[i,] = uu0[1]**2/np.sum(uu0[1]**2)
        psv = np.ones(k)
        for i in range(k):
            psv[i] = np.mean(dstat0[:,i] >= dstat[i])
        for i in range(1,k):
            psv[i] = np.max(psv[i-1], psv[i])
        nsv = np.sum(psv <= 0.1)
    #elif method == 'elbow':
    else:
        xraw = np.abs(np.diff(uu[1], n = 2))
        x = smooth(xraw, kind = '3RS3R', twiceit = True, endrule = 'Tukey')
        nsv = np.where(x<=np.quantile(x,0.5))[0][0]+1
        
    return nsv
            


def rotateSVD(svdres):
    upos = svdres[0]
    uneg = svdres[0]
    upos[upos<0] = 0
    uneg[uneg>0] = 0
    uneg = -uneg
    sumposu = upos.sum(axis = 0)
    sumnegu = uneg.sum(axis = 0)
    
    for i in range(svdres[1].shape[0]):
        if sumnegu[i] > sumposu[i]:
            svdres[0][:,i] = - svdres[0][:,i]
            svdres[2][i,] = -svdres[2][i,]
    return svdres


def round2(x):
    return round(x, 4)



    
    
def simpleDecomp(Y, k, svdres = None, L1 = None, L2 = None, max_iter= 500, tol = 5e-6, trace = False, rseed = None, B = None, scale = 1, adaptive_frac = 0.05, adaptive_iter = 30):
    
    pos_adj = 3
    ng, ns = Y.shape
    
    Bdiff = np.inf
    BdiffTrace = []
    BdiffCount = 0
    
    if svdres is None:
        svdres = randomized_svd(Y, n_components=k, n_iter = 3)
        svdres = rotateSVD(svdres)
    
    if L1 is None:
        L1 = svdres[1][k-1]*scale
        if pos_adj is not None:
            L1 = L1/pos_adj
        
    if L2 is None:
        L2 = svdres[1][k-1]*scale
    
    if B is None:
        B = (np.matmul(svdres[2][0:k,0:Y.shape[1]].T, np.diag(np.sqrt(svdres[1][0:k])))).T
        
    if rseed is not None:
        np.random.seed(rseed)
        B = [np.random.choice(B[el,],size = B.shape[1], replace = False) for el in range(B.shape[0])]
        B = np.array(B)
    
    def getT(x):
        nonlocal adaptive_frac
        return -np.quantile(x[x<0], adaptive_frac)

    for i in range(max_iter):
        #print(i)
        Z = np.matmul(np.matmul(Y, B.T), np.linalg.inv( np.matmul(B, B.T) +L1*np.diag( np.ones(k))) )
        Zraw = Z
        if (i >= adaptive_iter) & (adaptive_frac > 0):
            cutoffs = np.apply_along_axis(getT,0,Zraw)
        
            for j in range(Z.shape[1]):
                Z[Z[:,j]<cutoffs[j],j ] = 0
        else:
            Z[Z<0] = 0
        
        oldB = B
        B = np.matmul(np.linalg.inv( np.matmul(Z.T, Z)+ L2*np.diag(np.ones(k))) , np.matmul(Z.T, Y))
        
        Bdiff = np.sum( (B-oldB)**2)/np.sum(B**2)
        BdiffTrace.append(Bdiff)
        err0 = np.sum( (Y-np.matmul(Z,B))**2) + np.sum(Z**2)*L1+np.sum(B**2)*L2
        
        if trace:
            print(err0)
        #check for convergence
        if i > 52:
            if Bdiff > BdiffTrace[i-50]:
                BdiffCount = BdiffCount+1
        elif BdiffCount > 1:
            BdiffCount = BdiffCount-1
        
        if (Bdiff < tol) & (i>40):
            break
        elif BdiffCount >1:
            BdiffCount = BdiffCount-1
        
        if (Bdiff < tol) & (i > 40):
            break
        
        if (BdiffCount > 5) & (i > 40):
            break
    
    #rownames(B)=colnames(Z)
    Zproject = np.matmul(Z, np.linalg.inv(np.matmul(Z.T, Z)+L2*np.diag(np.ones(k))))
    return B, Z, Zraw, Zproject, L1, L2

