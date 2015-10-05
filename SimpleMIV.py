# -*- coding: utf-8 -*-
"""
Created on Mon Jun 09 14:49:14 2014

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np

# Mapping ImageJ TIFF data type id to numpy data type id
MIV_DTYPES = {
    0:  'B',    # byte
    1:  'h',    # 16bit signed
    2:  'H',    # 16bit unsigned
    3:  'i',    # 32bit signed
    4:  'f',    # 32bit float
    5:  'B',    # IndexedColor8bit
    6:  'B',    # RGB
    11: 'I',    # 32bit unsigned
    12: 'B',    # RGBA
    13: 'H',    # 12bit unsigned
    16: 'd'
}

MIV_RGB  = 6
MIV_RGBA = 12

def MIV_Load(filename):
    with open(filename, 'rb') as fp:
        if fp.read(4) != 'MIVF':
            return None
        xdim, ydim, zdim = vdim = np.fromfile(fp, np.int32, 3)
        if xdim <= 0 or ydim <= 0 or zdim <= 0:
            raise -2
        dt = np.fromfile(fp, np.int32, 1)[0]
        vt = MIV_DTYPES[dt]
        if dt == MIV_RGB:
            dm = (zdim,ydim,xdim,3)
        elif dt == MIV_RGBA:
            dm = (zdim,ydim,xdim,4)
        else:
            dm = (zdim,ydim,xdim)
        np.fromfile(fp, np.int32, 1)[0]
        np.fromfile(fp, np.float64, 3)
        np.fromfile(fp, np.int32, 1)
        np.fromfile(fp, np.uint8, 8)
        return np.fromfile(fp, np.dtype(vt)).reshape(dm), vdim

