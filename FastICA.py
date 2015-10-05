# -*- coding: utf-8 -*-
"""
Created on Sat Dec 07 16:57:33 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
#------------------------------------------------------------------------------
import numpy as np
from numpy import uint16, float64
from numpy import matrix, array, arange, diag, sqrt, power, ones
from numpy.linalg import eig
#------------------------------------------------------------------------------
# FastICA (symetric/pow3) implementation in pure Python/NumPy
#
# Usage:
#
#  >>> import FastICA
#  >>> Ux, Mx = FastICA.Execute(Data, umix=True)
#
#------------------------------------------------------------------------------
verbose = False

def ZeroMean(Sx):
    return Sx - matrix(ones(Sx.shape[0])).T * np.mean(Sx, 0)

def Whiten(ZMx):
    Ev, Et = eig(np.cov(ZMx.T) * ZMx.shape[1])
    WMx = Et.T * diag(1.0 / sqrt(Ev))
    DMx = diag(sqrt(Ev)) * Et
    return ZMx * WMx, WMx, DMx

def AdjustByGradients(Bx):
    Ev, Et = eig(Bx.T * Bx)
    return Bx * (Et * diag(1.0 / sqrt(Ev)) * Et.T)

def GenerateBx(nrIC, DMx):
    Bx = np.random.uniform(0.5, 1.0, nrIC)
    Tx = 1 - Bx
    Bx = np.diag(Bx)
    Bx[0,1] = Tx[1]
    Bx[1,0] = Tx[0]
    return Bx * np.linalg.inv(DMx)

def ShowProgress(n, Er, tuning):
    if not verbose:
        return
    print '.',
    if n % 40 == 0:
        print n

def Execute(Sx, niters=500, eps=1E-12, teps=1E-24, umix=False):
    nrSi, nrIC = Sx.shape
    Wx, WMX, DMX = Whiten(ZeroMean(Sx))
    Wt = Wx.T

    while True:
        Bx = GenerateBx(nrIC, DMX)
        Bx = AdjustByGradients(Bx)
        if np.isfinite(Bx).all():
            break

    Bf = 0.1
    tuning = False
    if nrSi <= 1E6:
        nrSx = nrSi
    else:
        nrSx = nrSi / (10 * nrIC)
    Em = 1E32
    for n in arange(1, niters+1):
        Bo = Bx.copy()
        if not tuning:
            Bx = (np.power(Bx * Wt, 3.0) * Wx) / nrSx - 3.0 * Bx
        else:
            Tx = Bx * Wt
            P3 = power(Tx, 3.0)
            Bt = np.sum(array(P3) * array(Tx), 1)
            Dx = diag(1.0 / (Bt - 3 * nrSx))
            Bx = Bx + Bf * (Bx * Dx * (P3 * Tx.T - diag(Bt)))
        Bx = AdjustByGradients(Bx)
        Er = 1.0 - np.min(np.abs(diag(Bx.T * Bo)))
        ShowProgress(n, Er, tuning)
        if Er < 0:
            if not tuning: # Retry
                break
            Bx = Bo
            Bf = Bf / 10
            if Bf < 1E-9:
                break
        elif Er < eps:
            if tuning:
                break
            if teps != None:
                if eps < teps:
                    break
                eps = teps
            tuning = True
        if Er > 0 and Er < Em:
            Em = Er
            Bm = Bx

    if Em < Er: # Restore Best Bx
        Bx = Bm

    if verbose:
        print '\nTotal number of iterations = %d, Er = %f' % (n, Er)

    Ax = DMX * Bx
    Wx = Bx.T * WMX
    if not umix:
        return Ax

    return Sx * Wx, Ax
