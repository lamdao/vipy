# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:15:23 2013

@author: Lam H. Dao  <daohailam(at)yahoo(dot)com>
"""

import gdcm
import numpy as np
from struct import pack

from VolUtils import Volume
from FileRoutines import XFile

# Mapping from DICOM data type ID to numpy data type ID
DIV_DTYPES = {
    gdcm.PixelFormat.UINT8:   'B',    # byte
    gdcm.PixelFormat.INT12:   'h',    # 12bit signed
    gdcm.PixelFormat.INT16:   'h',    # 16bit signed
    gdcm.PixelFormat.INT32:   'i',    # 32bit signed
    gdcm.PixelFormat.UINT12:  'H',    # 16bit unsigned
    gdcm.PixelFormat.UINT16:  'H',    # 16bit unsigned
    gdcm.PixelFormat.UINT32:  'I',    # 32bit unsigned
    gdcm.PixelFormat.FLOAT32: 'f',    # 32bit float
    gdcm.PixelFormat.FLOAT64: 'd',    # 64bit float (double)
}
# Mapping from numpy data type to ImageJ TIFF data type ID
DIV_VTYPES = {
    np.dtype(np.uint8)    : gdcm.PixelFormat.UINT8,
    np.dtype(np.int16)    : gdcm.PixelFormat.INT16,
    np.dtype(np.uint16)   : gdcm.PixelFormat.UINT16,
    np.dtype(np.int32)    : gdcm.PixelFormat.INT32,
    np.dtype(np.uint32)   : gdcm.PixelFormat.UINT32,
    np.dtype(np.float32)  : gdcm.PixelFormat.FLOAT32,
    np.dtype(np.float64)  : gdcm.PixelFormat.FLOAT64,
}
#-----------------------------------------------------------------------------
class DIVError(Exception):
    ERRORS = {
        -1: '%s is not DICOM folder.',
        -2: '%s is empty.',
        -3: '%s read error.'
    }

    def __init__(self, error_code, info = ''):
        self.message = DIVError.ERRORS[error_code]
        self.info = info
#-----------------------------------------------------------------------------
class DIV(Volume):
    """DICOM Image Volume (DIV)
    Load a serie of dicom files (*.dcm) into a 3D array
    """

    DEFAULT_EXTENSION = 'dcm'

    def __init__(self, source=None, dtype=None, extra=None):
        if isinstance(source,str):
            self.filename = source
            self.Load(source)
        else:
            super(DIV,self).__init__(source, extra)
        if self.data is not None and \
            dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        self.type = 1
        self.ExportDataInterfaces()

    def Load(self, dirname):
        try:
            if not XFile.PathExists(dirname):
                raise DIVError(-1)
            files = XFile.GetList(dirname, '*.'+DIV.DEFAULT_EXTENSION)
            if files is None or len(files) == 0:
                raise DIVError(-2)
            vol = None
            vdim = None
            for dfile in sorted(files):
                dicom = gdcm.ImageReader()
                dicom.SetFileName(dirname + '/' + dfile)
                if not dicom.Read():
                    raise DIVError(-3)
                image = dicom.GetImage()
                if vdim is None:
                    vdim = image.GetDimensions()    # Image dimensions
                    fmt  = image.GetPixelFormat()   # Pixel format
                    vs   = image.GetSpacing()       # Voxel size
                    vdim.append(len(files))
                    dt = DIV_DTYPES[fmt.GetScalarType()]
                raw = np.frombuffer(image.GetBuffer(), dtype=dt)
                vol = raw if vol is None else np.append(vol, raw)
            xdim, ydim, zdim = vdim
            vol = vol.reshape((zdim,ydim,xdim))
            vu = Volume.VOXEL_UNITS['mm'] 
            cl = [0]*8
            nc = 1
        except DIVError as err:
            print err.message % filename, err.info
            raise
        except Exception:
            raise
        else:
            self.data = vol
            self.xdim = xdim
            self.ydim = ydim
            self.zdim = zdim
            self.vdim = vdim
            self.vunit = vu
            self.vsize = vs
            self.nchan = nc
            self.clist = cl

    @staticmethod
    def Read(filename, info=True):
        try:
            m = DIV(filename)
            if info:
                return m.data, m.vdim, m.vsize, m.vunit
            return m.data
        except Exception:
            raise
        return None
