# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:15:23 2013

@author: Lam H. Dao  <daohailam(at)yahoo(dot)com>
"""

import numpy as np
from struct import pack

from VolUtils import Volume
from FileRoutines import XFile

# Mapping from RAW data type ID to numpy data type ID
RIV_DTYPES = {
    'b':  'B',    # byte
    's':  'h',    # 16bit signed
    'u':  'H',    # 16bit unsigned
    'i':  'i',    # 32bit signed
    'n':  'I',    # 32bit unsigned
    'f':  'f',    # 32bit float
    'd':  'd',    # 64bit float (double)
    'c':  'B'     # RGB
}
# Mapping from numpy data type to ImageJ TIFF data type ID
RIV_VTYPES = {
    np.dtype(np.uint8)    : 'b',
    np.dtype(np.int16)    : 's',
    np.dtype(np.uint16)   : 'u',
    np.dtype(np.int32)    : 'i',
    np.dtype(np.uint32)   : 'n',
    np.dtype(np.float32)  : 'f',
    np.dtype(np.float64)  : 'd',
}
#-----------------------------------------------------------------------------
class RIVError(Exception):
    ERRORS = {
        -1: '%s is not RAW/RIV file.',
        -2: '%s has incorrect dimensions: '
    }

    def __init__(self, error_code, info = ''):
        self.message = RAWError.ERRORS[error_code]
        self.info = info
#-----------------------------------------------------------------------------
class RIV(Volume):
    """RAW Image Volume (RIV/RAW)

    RAW file = Raw data stored linearly +
               Meta info (dimensions, datatype, voxel size/unit, ...) is
               formated in filename (e.g. stack_1_1.512x512x256b.raw)
    """

    DEFAULT_EXTENSION = 'raw'

    def __init__(self, source=None, dtype=None, extra=None):
        if isinstance(source,str):
            self.filename = source
            self.Load(source)
        else:
            super(RIV,self).__init__(source, extra)
        if self.data is not None and \
            dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        self.type = 1
        self.ExportDataInterfaces()

    def Load(self, filename):
        try:
            if XFile.GetExt(filename) != RIV.DEFAULT_EXTENSION:
                raise RIVError(-1)
            fn = XFile.CutExt(filename) # filename without ext
            mi = XFile.GetExt(fn)       # extract metainfo
            if mi == '':
                raise RIVError(-1)
            dt = mi[-1]                 # last char in metainfo is datatype
            vt = RIV_DTYPES['b']        # assume 'byte' type if no datatype
            if RIV_DTYPES.has_key(dt):
                vt = RIV_DTYPES[dt]
                mi = mi[:-1]
            mi = mi.split('x')
            if len(mi) != 3:
                raise RIVError(-2, mi)
            xdim, ydim, zdim = vdim = [int(v) for v in mi]
            if xdim <= 0 or ydim <= 0 or zdim <= 0:
                raise RIVError(-2, vdim)
            # Load data type ID (compatible with ImageJ data type)
            dm = (zdim,ydim,xdim) if dt != 'c' else (zdim,ydim,xdim,3)
            # Voxel unit (See Volume.VOXEL_UNITS)
            vu = 0                      # no voxel unit (default is pixel)
            # Voxel size
            vs = [1.0]*3                # no voxel size (default [1,1,1])
            # Number of channels
            nc = 1                      # only support 1 channel
            # Color map id (support maximum 8 channels)
            cl = [0]*8                  # color table is gray
            with open(filename, 'rb') as fp:
                # Load data
                vb = np.fromfile(fp, np.dtype(vt)).reshape(dm)
        except RIVError as err:
            print err.message % filename, err.info
            raise
        except Exception:
            raise
        else:
            self.data = vb
            self.xdim = xdim
            self.ydim = ydim
            self.zdim = zdim
            self.vdim = vdim
            self.vunit = vu
            self.vsize = vs
            self.nchan = nc
            self.clist = cl

    def Save(self, filename=None):
        """Save(filename) -- save data to file: filename.<dim><dtype>.raw
        dim   - dimensions
        dtype - datatype

        these information and extension (.raw) will be added automatically
        using internal information
        """

        if isinstance(filename, str):
            self.filename = filename
        while self.filename == None or self.filename == '':
            self.filename = raw_input('Enter new filename for RAW: ').strip()
        vt = RIV_VTYPES[self.data.dtype]
        mi = '.%dx%dx%d%s.' % (self.xdim,self.ydim,self.zdim,vt)
        self.filename += mi + RIV.DEFAULT_EXTENSION

        print 'Saving data to %s ... ' % (self.filename),
        with open(self.filename, 'wb') as fp:
            self.data.tofile(fp)
        print 'done.'

    @staticmethod
    def Read(filename, info=True):
        try:
            m = RIV(filename)
            if info:
                return m.data, m.vdim, m.vsize, m.vunit
            return m.data
        except Exception:
            raise
        return None
