# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:15:23 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np
from struct import pack

from VolUtils import Volume
from FileRoutines import XFile

# Mapping from ImageJ TIFF data type ID to numpy data type ID
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
# Mapping from numpy data type to ImageJ TIFF data type ID
MIV_VTYPES = {
    np.dtype(np.uint8)    : 0,
    np.dtype(np.int16)    : 1,
    np.dtype(np.uint16)   : 2,
    np.dtype(np.int32)    : 3,
    np.dtype(np.float32)  : 4,
    np.dtype(np.float64)  : 16,
}
#-----------------------------------------------------------------------------
MIV_RGB  = 6
MIV_RGBA = 12
#-----------------------------------------------------------------------------
class MIVError(Exception):
    ERRORS = {
        -1: '%s is not MIV file.',
        -2: '%s has incorrect dimensions: '
    }

    def __init__(self, error_code, info = ''):
        self.message = MIVError.ERRORS[error_code]
        self.info = info
#-----------------------------------------------------------------------------
class MIV(Volume):
    """Microscopic Image Volume (MIV)

    MIV file = Raw data stored linearly +
               Meta info (dimensions, datatype, voxel size/unit, ...)

    This file format allows to access volume data linearly, while TIFF file
    allows access data as slice by slice. It is faster in loading, saving data
    when using MIV format. Besides, it also allows using memory mapping to
    access file directly as a linear data array without loading into memory.

    This format was designed to work with microscope image stacks. However, it
    can be used to store other volume data types (e.g. TIFF, DICOM), especially
    in computation/processing duration before exporting back to original format.
    """

    DEFAULT_EXTENSION = 'miv'

    def __init__(self, source=None, dtype=None, extra=None):
        if isinstance(source,str):
            self.filename = source
            self.Load(source)
        else:
            super(MIV,self).__init__(source, extra)
        if self.data is not None and \
            dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        self.type = 1
        self.ExportDataInterfaces()

    def Load(self, filename):
        try:
            with open(filename, 'rb') as fp:
                # Check signature
                if fp.read(4) != 'MIVF':
                    raise MIVError(-1)
                # Load/check dimensions 
                xdim, ydim, zdim = vdim = np.fromfile(fp, np.int32, 3)
                if xdim <= 0 or ydim <= 0 or zdim <= 0:
                    raise MIVError(-2, vdim)
                # Load data type ID (compatible with ImageJ data type)
                dt = np.fromfile(fp, np.int32, 1)[0]
                vt = MIV_DTYPES[dt]
                if dt == MIV_RGB:
                    dm = (zdim,ydim,xdim,3)
                elif dt == MIV_RGBA:
                    dm = (zdim,ydim,xdim,4)
                else:
                    dm = (zdim,ydim,xdim)
                # Voxel unit (See Volume.VOXEL_UNITS)
                vu = np.fromfile(fp, np.int32, 1)[0]
                # Voxel size
                vs = np.fromfile(fp, np.float64, 3)
                # Number of channels
                nc = np.fromfile(fp, np.int32, 1)
                # Color map id (support maximum 8 channels)
                cl = np.fromfile(fp, np.uint8, 8)
                # Load data
                vb = np.fromfile(fp, np.dtype(vt)).reshape(dm)
        except MIVError as err:
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
        if isinstance(filename, str):
            self.filename = filename
        while self.filename == None or self.filename == '':
            self.filename = raw_input('Enter new filename: ').strip()
            if self.filename != '' and XFile.GetExt(self.filename) != 'miv':
                self.filename = XFile.NewExt(self.filename, 'miv')
        print 'Saving data to %s ... ' % (self.filename),
        MIV.Write(self.filename, self.data, self.vsize, self.vunit, self.clist)
        print 'done.'

    @staticmethod
    def Read(filename, info=True):
        try:
            m = MIV(filename)
            if info:
                return m.data, m.vdim, m.vsize, m.vunit
            return m.data
        except Exception:
            raise
        return None

    @staticmethod
    def Write(filename, data, voxel_size=(1.,1.,1.), unit=0, clist=None):
        if isinstance(unit, basestring):
            try:
                vu = Volume.VOXEL_UNITS[unit]
            except Exception:
                vu = 0
        elif isinstance(unit, (int,long)) and unit >= 0:
            vu = unit
        else:
            vu = 0
        vt = MIV_VTYPES[data.dtype]
        vd = list(data.shape)[::-1]
        if len(vd) == 4:
            vc = vd[0]
            vd = vd[1:4]
            if data.dtype == np.uint8 and vc in [3,4]:
                vt = {3:MIV_RGB,4:MIV_RGBA}[vc]
                vc = 0
        else:
            vc = 0
        if vc == 0:
            clist = np.zeros(8, dtype=np.uint8)
        elif clist == None:
            clist = np.arange(0, 8, dtype=np.uint8)
        elif len(clist) < 8:
            cl = [0] * 8
            cl[0:len(clist)] = clist
            clist = np.array(cl, dtype=np.uint8)
        vs = list(voxel_size)
        with open(filename, 'wb') as fp:
            fp.write('MIVF')
            fp.write(pack('3i',*vd))
            fp.write(pack('2i3d',vt,vu,*vs))
            fp.write(pack('i',vc))
            clist.tofile(fp)
            if vc == 0:
                data.tofile(fp)
            else:
                for c in range(vc):
                    data[:,:,:,c].tofile(fp)
