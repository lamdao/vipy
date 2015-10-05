# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:23:34 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
from ExtLib import ExternalLib

from ctypes import c_int32, byref

import numpy as np

class VolFilters(ExternalLib):
    """This is a wrapper class for VolFilters.{dll,so,dylib} written in C/C++
    """
    def __init__(self):
        ExternalLib.Init(self, 'VolFilters')

    def PrepareKernel(self, kernel):
        if isinstance(kernel, int):
            ksizes = np.array([1, kernel], dtype=int)
            return ksizes, ksizes
        return kernel, kernel.cdim

    def Median(self, volume, kernel):
        kernel, ksizes = self.PrepareKernel(kernel)
        result = volume.Clone(data=[])
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, result, ksizes, kernel, byref(ncpu))
        self.Execute(self.lib.vol_median_filter)
        return result

    def Mean(self, volume, kernel):
        kernel, ksizes = self.PrepareKernel(kernel)
        result = volume.Clone(data=[])
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, result, ksizes, kernel, byref(ncpu))
        self.Execute(self.lib.vol_mean_filter)
        return result

    def Sigma(self, volume, kernel, params=[2.0, 0.2, 255.0]):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        kernel, ksizes = self.PrepareKernel(kernel)
        result = volume.Clone(data=[])
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, result, ksizes, kernel, params, byref(ncpu))
        self.Execute(self.lib.vol_sigma_filter)
        return result

    def Anisotropic(self, volume, params=[180.0, 1.0/14, 10]):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        result = volume.AsType(float)
        rsizes = result.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(rsizes, result, result, params, byref(ncpu))
        self.Execute(self.lib.vol_anisotropic_filter)
        return result

    def Laplace(self, volume, params=[0.0, 0.05, 50]):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        result = volume.AsType(float)
        rsizes = result.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(rsizes, result, result, params, byref(ncpu))
        self.Execute(self.lib.vol_laplace_filter)
        return result

    def Convol(self, volume, kernel):
        ncpu = c_int32(self.npt)
        vsizes = volume.cdim
        self.AddParameters(vsizes, volume, kernel, byref(ncpu))
        self.Execute(self.lib.vol_convol)
        return np.fft.ifftshift(volume.data)

    def FrequencyMask(self, volume, mask):
        ncpu = c_int32(self.npt)
        vsizes = volume.cdim
        self.AddParameters(vsizes, volume, mask, byref(ncpu))
        self.Execute(self.lib.vol_freq_mask)
        return volume

    def MedianCDF(self, volume, kernel):
        kernel, ksizes = self.PrepareKernel(kernel)
        result = volume.Clone(data=[])
        vsizes = volume.cdim
        ncpu = c_int32(self.npt)
        self.AddParameters(vsizes, volume, result, ksizes, kernel, byref(ncpu))
        self.Execute(self.lib.vol_mediancdf_filter)
        return result

