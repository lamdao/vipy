# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:24:15 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np
from ExtLib import ExternalLib

class VolMorphology(ExternalLib):
    """This is a wrapper class for VolMorphology.{dll,so,dylib} written in C/C++
    """
    def __init__(self):
        ExternalLib.Init(self, 'VolMorphology')

    def SetupParameters(self, Image, Structure, Gray=None):
        p = list(Image.shape)
        p.extend([Structure.shape[0], self.npt, bool(Gray)])
        self.Params = np.array(p, dtype=np.int32)
        self.Kernel = Structure.astype(np.uint16, copy=True)
        self.Result = Image.astype(np.uint16, copy=True)
        self.AddParameters(self.Result, self.Kernel, self.Params)

    def Dilate(self, Image, Structure, Gray=None):
        self.SetupParameters(Image, Structure, Gray)
        self.Execute(self.lib.p_morph_dilate)
        return self.Result

    def Erode(self, Image, Structure, Gray=None):
        self.SetupParameters(Image, Structure, Gray)
        self.Execute(self.lib.p_morph_erode)
        return self.Result

    def Open(self, Image, Structure, Gray=None):
        self.SetupParameters(Image, Structure, Gray)
        self.Execute(self.lib.p_morph_open)
        return self.Result

    def Close(self, Image, Structure, Gray=None):
        self.SetupParameters(Image, Structure, Gray)
        self.Execute(self.lib.p_morph_close)
        return self.Result

    def TopHat(self, Image, Structure, Gray=None):
        return Image - self.Open(Image, Structure, Gray)

    instance = None

    @classmethod
    def Create(clz):
        if clz.instance == None:
            clz.instance = VolMorphology()
        return clz.instance

