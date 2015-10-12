# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:33:27 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import os
import platform
import numpy as np
from multiprocessing import cpu_count
from ctypes import c_voidp, c_int32, cdll, cast

from FileRoutines import XFile

class ExternalLib(object):
    DEFAULT_SHARED_LIB_PATH = os.getenv('SHARED_LIBS')

    def __init__(self, dll):
        ExternalLib.Init(self, dll)

    def ResetParameters(self):
        self.args = []

    def AddParameter(self, value):
        if isinstance(value, np.ndarray):
            self.args.append(cast(np.ctypeslib.as_ctypes(value), c_voidp))
        elif value.__class__.__name__ in ['Volume', 'MIV', 'TIV']:
            self.args.append(cast(np.ctypeslib.as_ctypes(value.data), c_voidp))
        else:
            self.args.append(cast(value, c_voidp))

    def AddParameters(self, *values):
        self.ResetParameters()
        for p in values:
            self.AddParameter(p)

    def Execute(self, Fx):
        args = (c_voidp * len(self.args))(*self.args)
        Fx.argtypes = [c_int32, c_voidp]
        Fx(len(args), args)

    @staticmethod
    def GetFullPath(dll):
        if XFile.Exists(dll):
            return os.path.abspath(dll)
        dpath = ExternalLib.DEFAULT_SHARED_LIB_PATH
        if dpath is None:
            dpath = os.path.abspath('./lib')
            if not XFile.PathExists(dpath):
                return dll
        return dpath + os.sep + dll

    @staticmethod
    def Init(self, dll):
        if XFile.GetExt(dll) == '':
            if platform.system() == 'Windows':
                dll += '.dll'
            else:
                dll += '.so'

        try:
            self.lib = cdll.__getattr__(XFile.CutExt(dll))
        except Exception:
            self.lib = cdll.LoadLibrary(self.GetFullPath(dll))

        self.dll = dll
        if platform.system() == 'Windows':
            self.DFree = cdll.kernel32.FreeLibrary
            self.hid = c_voidp(self.lib._handle)
        else:
            self.DFree = cdll.LoadLibrary('libdl.so').dlclose
            self.hid = self.lib._handle
        self.npt = cpu_count()
        self.args = []

    def Release(self):
        try:
            while self.DFree(self.hid) == 1:
                pass
            cdll.__delattr__(XFile.CutExt(self.dll))
        except Exception:
            pass

