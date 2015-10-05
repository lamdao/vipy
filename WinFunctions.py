# -*- coding: utf-8 -*-
"""
Created on Thu Jan 09 11:51:10 2014

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np
from StdFunctions import StdFx
#------------------------------------------------------------------------------
class WinFx(object):
    @staticmethod
    def Gaussian(sigma, npts=None):
        if npts == None:
            npts = 2 * (np.ceil(sigma * 3).astype(np.int32) | 1) + 1
        if isinstance(npts,(int,long,np.int32,np.int64)):
            npts = np.arange(npts, dtype=float) - npts // 2
        return np.exp(-npts**2.0 / (2*sigma**2.0))
    @staticmethod
    def Exponential(npts):
        x = np.arange(npts // 2, dtype=float) / (npts - 1)
        w = 1.0 - 1.0 / np.power(1000.0, x * 2)
        if (npts & 1) == 1:
            c = (npts//2 + 1.0) / (npts - 1)
            c = 1.0 - 1.0 / np.power(1000.0, c * 2)
            return np.concatenate((w,[c],w[::-1]))
        return np.sqrt(np.concatenate((w,w[::-1])))
    @staticmethod
    def Hanning(npts):
        return np.hanning(npts)
    @staticmethod
    def Hamming(npts):
        return np.hamming(npts)
    @staticmethod
    def Sinc(npts, n=3):
        if npts == 1:
            return np.array([1.0])
        x = 2 * n * (np.arange(npts, dtype=float) / (npts-1) - 0.5)
        return StdFx.Sinc(x)
    @staticmethod
    def Lanczos(npts, n=3):
        if npts == 1:
            return np.array([1.0])
        x = 2 * n * (np.arange(npts, dtype=float) / (npts-1) - 0.5)
        return StdFx.Sinc(x) * StdFx.Sinc(x/n)
#------------------------------------------------------------------------------
