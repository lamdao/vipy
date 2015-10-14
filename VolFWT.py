# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:23:34 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
from ExtLib import ExternalLib

from ctypes import c_int32, byref

import numpy as np

class VolFWT(ExternalLib):

    def __init__(self, wavelet=None):
        ExternalLib.Init(self, 'VolFWT')
        self.SetWavelet(wavelet)

    def SetWavelet(self, source):
        if isinstance(source, str):
            source = np.genfromtxt(source)
        if not isinstance(source, np.ndarray):
            return
        self.AddParameters(source)
        self.Execute(self.lib.set_wavelet)

    def FWT1D(self, buf, levels=1, forward=True):
        src = buf
        dst = np.empty_like(src)
        dim = np.array([len(src), levels], dtype=np.int32)
        self.AddParameters(dim, src, dst)
        self.Execute(self.lib.fwt1d)
        return dst

    def FWT2D(self, img, levels=1, forward=True):
        src = img.AsType(float)
        dst = src.Clone(data=0)
        dim = src.cdim
        pwt = np.array([forward, levels], dtype=np.int32)
        ncpu = c_int32(self.npt)
        self.AddParameters(dim, src, dst, pwt, byref(ncpu))
        self.Execute(self.lib.fwt2d)
        return src

    def FWT3D(self, vol, levels=1, forward=True):
        src = vol.AsType(float)
        dst = src.Clone(data=[])
        dim = src.cdim
        pwt = np.array([forward, levels], dtype=np.int32)
        ncpu = c_int32(self.npt)
        self.AddParameters(dim, src, dst, pwt, byref(ncpu))
        self.Execute(self.lib.fwt3d)
        return dst
