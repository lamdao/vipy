# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:15:23 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np
from tifffile import TiffFile
from struct import pack
from VolUtils import Volume

#-----------------------------------------------------------------------------
# Tiff Image Volume
#-----------------------------------------------------------------------------
class TIV(Volume):

    DEFAULT_EXTENSION = 'tif'

    def __init__(self, filename, dtype=None):
        self.type = 2
        tf = TiffFile(filename)
        vb = tf.asarray()

        nc = 0
        if len(vb.shape) == 3:
            zdim, ydim, xdim = vb.shape
        else:
            zdim, ydim, xdim, nc = vb.shape

        if nc > 0 and vb.dtype == np.uint8:
            nc = 0

        dp = tf.pages[0]
        try:
            xs = float(dp.x_resolution[1]) / dp.x_resolution[0]
            ys = float(dp.y_resolution[1]) / dp.y_resolution[0]
        except Exception:
            xs = ys = 1.0
        zs = 1.0
        vu = 0
        if dp.is_imagej:
            if dp.imagej_tags.has_key('spacing'):
                zs = float(dp.imagej_tags.spacing)
            if dp.imagej_tags.has_key('unit') and dp.imagej_tags.unit != '':
                vu = Volume.VOXEL_UNITS[dp.imagej_tags.unit]
        vs = [xs, ys, zs]

        self.filename = filename
        self.data = vb if dtype is None or dtype is vb.dtype else vb.astype(dtype)
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.vdim = np.array([xdim, ydim, zdim])
        self.vunit = vu
        self.vsize = np.round(vs, 3)
        self.nchan = nc
        self.clist = [0] * 8

    @staticmethod
    def Read(filename, info=True):
        try:
            m = TIV(filename)
            if info:
                return m.data, m.vdim, m.vsize, m.vunit
            return m.data
        except Exception:
            raise
        return None

