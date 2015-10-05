# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:23:34 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
from ExtLib import ExternalLib
from ctypes import c_int32, byref
from numpy import array, ceil
import numpy as np

class VolProcess(ExternalLib):
    """This is a wrapper class for VolProcess.{dll,so,dylib} written in C/C++
    """
    RESCALE_METHOD_ID = {
        'nearest':  0,
        'linear':   1,
        'cubic':    2,
        'lanczos':  3,
        'akima':    4
    }
    def __init__(self):
        ExternalLib.Init(self, 'VolProcess')

    def Rescale(self, volume, scales, mode='linear', rsize=None):
        dim = ceil(volume.vdim * scales)
        if rsize is not None:
            dim = rsize
        result = volume.Clone(data=dim[::-1])
        vsizes = volume.cdim
        rsizes = result.cdim
        ncpu = c_int32(self.npt)
        mode = c_int32(self.RESCALE_METHOD_ID[mode])

        self.AddParameters(rsizes, result, vsizes, volume, byref(ncpu), byref(mode))
        self.Execute(self.lib.vol_rescale)
        return result

    def Mirror(self, volume, ext=None, fftsize=False):
        dim = volume.vdim
        if ext is None:
            ext = dim // 4
        else:
            ext = array(ext)
        dim = dim + ext + ((ext % 2) != 0)
        if fftsize:
            dim = volume.FFTSize(dim)
        result = volume.Clone(data=dim[::-1])
        vsizes = volume.cdim
        rsizes = result.cdim
        ncpu = c_int32(self.npt)

        self.AddParameters(vsizes, rsizes, volume, result, byref(ncpu))
        self.Execute(self.lib.vol_mirror)
        return result

    def CenterDistance(self, volume, nosqrt=False, center=True):
        params = array([nosqrt, center], dtype=np.int32)
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, params, byref(ncpu))
        self.Execute(self.lib.vol_cdistance)

    def Homomorphic(self, volume, D0, GH, GL, C, nosqrt=False, center=True):
        parameters = array([D0, GH, GL, C], dtype=np.float64)
        options = array([nosqrt, center], dtype=np.int32)
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, options, parameters, byref(ncpu))
        self.Execute(self.lib.vol_homomorphic)

