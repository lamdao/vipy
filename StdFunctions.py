# -*- coding: utf-8 -*-
"""
Created on Thu Jan 09 11:49:21 2014

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np
import matplotlib
#------------------------------------------------------------------------------
class StdFx(object):
    @staticmethod
    def Unit(npts, sym=False):
        s = 0.0
        if sym:
            s = 0.5
        u = np.arange(npts) / float(npts-1)
        return u - s
    @staticmethod
    def Gaussian(sigma, npts=None):
        if npts == None:
            npts = 2 * (np.ceil(sigma * 3).astype(np.int32) | 1) + 1
        if isinstance(npts,(int,long,np.int32,np.int64)):
            npts = np.arange(npts, dtype=float) - npts // 2
        return np.exp(-npts**2.0 / (2*sigma**2.0))
    @staticmethod
    def Sinc0(x):
        return 1.0 - x**2/6.0 + x**4/120.0
    @staticmethod
    def Sinc(x):
        x = np.pi * x
        r = np.sin(x) / x
        if isinstance(x,np.ndarray):
            n = np.where(np.isnan(r))
            r[n] = StdFx.Sinc0(x[n])
        elif np.isnan(r):
            r = StdFx.Sinc0(x)
        return r
    @staticmethod
    def Lanczos(x, n=3):
        if n <= 0:
            return np.array([])
        return StdFx.Sinc(x) * StdFx.Sinc(x/n)
    @staticmethod
    def Gabor(x, sigma=1.0, T=1.0):
        s = 2.0 * (sigma**2)
        p = 2.0 * np.pi / T
        return np.exp(-x**2.0/s) * np.cos(p*x)

    @staticmethod
    def LogFact(n):
        t = n + 1.0
        return (t - 0.5) * np.log(t) - t + 0.918938533204672670 + \
                1.0 / (12.0 * t) - \
                1.0 / (360.0 * (t ** 3.0)) + \
                1.0 / (1260.0 * (t ** 5.0))

    @staticmethod
    def Poisson(x, mu):
        xt = x / 10.0
        mt = mu / 10.0
        #y = np.exp(x * np.log(mu) - StdFx.LogFact(x) - mu)
        yt = np.exp(xt * np.log(mt) - StdFx.LogFact(xt) - mt)
        return yt / yt.max()

    @staticmethod
    def Poisson3(x, mu, shift=[10,55]):
        y = StdFx.Poisson(x, mu - shift[0]) + \
            StdFx.Poisson(x, mu) + \
            StdFx.Poisson(x, mu + shift[1])
        return y / y.max()

    @staticmethod
    def Weibull(x, L=None, k=2.0):
        if L is None: L = x.size / 2.0
        return np.exp(-((x.astype(np.float64)/L) ** k) * np.log(2.0))
    
    def Hill(x, L=None, k=2):
        if L is None: L = x.size / 2.0
        return 1.0 / (1.0 + (x.astype(np.float64)/L)**k)

    @staticmethod
    def Plot(x, y, XRANGE=None, YRANGE=None):
        if y is None:
            y = x
            x = np.arange(y.size)
        matplotlib.pyplot.plot(x, y)
        if XRANGE is None: XRANGE=[0,y.size-1]
        if YRANGE is None: YRANGE=[y.min(),y.max()]
        matplotlib.pyplot.xlim(XRANGE[0],XRANGE[1])
        matplotlib.pyplot.ylim(YRANGE[0],YRANGE[1])
#------------------------------------------------------------------------------
