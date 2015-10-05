# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:21:17 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

from MIVFile import MIV
from TIVFile import TIV
from FileRoutines import XFile

def Tiff2MIV(sdir, odir, pattern='*.tif'):
    if not XFile.PathExists(odir):
        XFile.MkDir(odir)

    for f in XFile.GetList(sdir, pattern):
        print 'Loading', f, '...',
        v = TIV(sdir + '/' + f)
        print 'done.'
        v = MIV(v.data[:,:,:,0].copy())
        v.Save(odir + '/' + XFile.NewExt(f, 'miv'))

if __name__ == '__main__':
    Tiff2MIV('D:/Data/ExVivo3', 'D:/Data/4x4', 'stack_*.tif')
