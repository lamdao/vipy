# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:20:00 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import os
import glob
import numpy as np

class XFile(object):
    """A simple class for file/folder manipulation"""

    @staticmethod
    def Path(fn):
        """Return folder name of file [fn]"""
        return os.path.dirname(fn)

    @staticmethod
    def Name(fn):
        """Return file name without path of file [fn]"""
        return os.path.basename(fn)

    @staticmethod
    def Base(fn):
        """Return file name without path and extension of file [fn]"""
        return XFile.BareName(fn)

    @staticmethod
    def BareName(fn):
        """Return file name without path and extension of file [fn]"""
        fn = XFile.Name(fn)
        p = fn.rfind('.')
        return fn[:p] if p >= 0 else fn

    @staticmethod
    def Ext(fn):
        """Return file extension of file [fn]"""
        return XFile.GetExt(fn)

    @staticmethod
    def GetExt(fn):
        """Return file extension of file [fn]"""
        p = fn.rfind('.')
        return fn[p+1:] if p >= 0 else ''

    @staticmethod
    def CutExt(fn):
        """Return file path+name without file extension of file [fn]"""
        p = fn.rfind('.')
        return fn[:p] if p >= 0 else fn

    @staticmethod
    def NewExt(fn, ext):
        """Return new file name with new extension of file [fn]"""
        return XFile.CutExt(fn) + '.' + ext

    @staticmethod
    def Exists(fn):
        """Check for the existence of file [fn]"""
        return bool(fn) and os.path.isfile(fn)

    @staticmethod
    def GetList(path, pattern='*'):
        """Return a list of files in [path] that match [pattern]"""
        sp = os.getcwd()
        os.chdir(path)
        lf = glob.glob(pattern)
        os.chdir(sp)
        return lf

    @staticmethod
    def PathExists(dname):
        """Check for the existence of folder [dname]"""
        return bool(dname) and os.path.isdir(dname)

    @staticmethod
    def MakeDir(dname):
        """Create new folder named [dname]"""
        try:
            os.mkdir(dname)
        except:
            pass

    @staticmethod
    def LoadCSV(fn):
        """Return contents of CSV file [fn] as numpy array"""
        return np.genfromtxt(fn, delimiter=',')

def File_GetPath(fn):
    return XFile.Path(fn)

def File_GetName(fn):
    return XFile.Name(fn)

def File_GetBase(fn):
    return XFile.BareName(fn)

def File_GetBareName(fn):
    return XFile.BareName(fn)

def File_GetExt(fn):
    return XFile.GetExt(fn)

def File_CutExt(fn):
    return XFile.CutExt(fn)

def File_NewExt(fn, ext):
    return XFile.NewExt(fn, ext)

def File_Exists(fn):
    return XFile.Exists(fn)

def File_GetList(path, pattern='*'):
    return XFile.GetList(path, pattern)

def Path_Exists(dname):
    return XFile.PathExists(dname)

def LoadCSV(fn):
    return XFile.LoadCSV(fn)

