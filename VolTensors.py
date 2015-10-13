# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:23:34 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
from ExtLib import ExternalLib

from ctypes import c_int32, byref

import numpy as np

class VolTensors(ExternalLib):
    """This is a wrapper class for VolTensors.{dll,so,dylib} written in C/C++
    """

    def __init__(self):
        ExternalLib.Init(self, 'VolTensors')

    def VolTensors(self, vol, EigenVectors=False):
        src = vol.AsType(np.float32)
        edv = src.Clone(data=np.ndarray(vol.shape+(3,),dtype=src.dtype))
        evx = None
        if EigenVectors is True:
            evx = src.Clone(data=np.ndarray(vol.shape+(9,),dtype=src.dtype))
        dim = src.cdim
        self.AddParameters(dim, src, edv, evx)
        self.Execute(self.lib.vol_calc_tensors)
        if EigenVectors is True:
            return edv, evx
        return edv
