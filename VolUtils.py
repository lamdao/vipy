# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:20:53 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
#------------------------------------------------------------------------------
import numpy as np
import numexpr as nx
from numpy import uint8, int32
from numpy import ceil, log
from numpy import array, ndarray
from numpy.matrixlib import matrix
from numpy.lib import hanning
#------------------------------------------------------------------------------
from FileRoutines import XFile
from VolFilters import VolFilters
from VolProcess import VolProcess
from Morphology import VolMorphology
from WinFunctions import WinFx
#------------------------------------------------------------------------------
def CalcFFTSize(d, p2=False):
    if d < 128 or p2:
        return 2**ceil(log(d) / log(2.0))
    n = 1
    while (1 << n) < d: n = n + 1
    n = 1 << (n-1)
    return long(n + CalcFFTSize(d - n))
#------------------------------------------------------------------------------
def CalcGaussianSigma(npts):
    n = int(npts) // 2
    if (n & 1) == 0:
        n -= 1
    return float(n) / 3
#------------------------------------------------------------------------------
class VolumeController(object):
    def __init__(self, clz):
        self.clz = clz
    def __get__(self, instance, outerclass):
        class Wrapper(self.clz):
            vol = instance
        Wrapper.__name__ = self.clz.__name__
        return Wrapper
#------------------------------------------------------------------------------
class Volume(object):
    DTYPE_VMAX = {
        np.uint8:   255.0,
        np.int16:   4096.0,
        np.uint16:  65535.0,
        np.int32:   2147483647.0,
        np.uint32:  2147483647.0,
        np.float32: 2147483647.0,
        np.float64: 2147483647.0,
    }

    DTYPE_CID = {
        np.uint8    : 1,
        np.int16    : 2,
        np.uint16   : 12,
        np.int32    : 3,
        np.uint32   : 13,
        np.float32  : 4,
        np.float64  : 5,
    }

    VOXEL_UNITS = {
        "pixel"     : 0,
        "nanometer" : 5,
        "micron"    : 6,
        "um"        : 6,
        "mm"        : 7,
        "cm"        : 8,
        "meter"     : 9,
        "km"        : 10,
        "inch"      : 11,
        "ft"        : 12,
        "mi"        : 13,
    }

    def __init__(self, source=None, extra=None):
        self.InitAttributes()
        if isinstance(source, Volume):
            self.CopyAttributes(source)
            if isinstance(extra, (tuple,list,np.ndarray)):
                self.xdim, self.ydim, self.zdim = self.vdim = source.vdim + extra
                bp = extra // 2; ep = bp + source.vdim
                self.data = np.ndarray(self.vdim[::-1], dtype=source.data.dtype)
                self.data[bp[2]:ep[2],bp[1]:ep[1],bp[0]:ep[0]] = source.data
            else:
                self.xdim, self.ydim, self.zdim = self.vdim = source.vdim
                self.data = source.data.copy()
        elif isinstance(source, ndarray):
            self.zdim, self.ydim, self.xdim = source.shape[0:3]
            self.vdim = array([self.xdim, self.ydim, self.zdim])
            self.data = source
        elif isinstance(source, tuple):
            self.xdim, self.ydim, self.zdim = self.vdim = array(source)
            self.data = ndarray(self.vdim[::-1], extra if extra is not None else uint8)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    @property
    def shape(self):
        try:
            return self.zdim, self.ydim, self.xdim
        except Exception:
            pass
        return (0,0,0)
    @property
    def psize(self):
        return self.xdim * self.ydim

    @property
    def flat(self):
        return self.data.ravel()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def cdim(self):
        w, h, d = self.vdim
        s = self.data.size
        t = Volume.DTYPE_CID[self.dtype.type]
        return np.array([3, w, h, d, t, s], dtype=np.uint32)

    def Locate(self, val, getmean=False):
        w = self.xdim
        p = w * self.ydim
        n = np.where(self.data.ravel() == val)[0]
        z = n // p; n = n % p
        y = n // w; x = n % w
        if getmean:
            return x.mean(), y.mean(), z.mean()
        return x, y, z

    def ExportDataInterfaces(self):
        if self.data is not None:
            self.Max = self.data.max
            self.Min = self.data.min
            self.DataRange = self.data.ptp

#
#   Data range manipulation
#
    def CutOff(self, value, tp=None):
        """Cast all value below a certain value to 0
        - value: the threshold value
        - tp: threshold in percentage (default: None)
        """
        if self.data is not None:
            if isinstance(tp, float) and 0.0 <= tp <= 1.0:
                value = self.data.max() * tp
            r = self.Clone()
            r.data[r.data <= value] = 0
            return r
        return self.Clone()
    
#
#   Attributes methods
#
    def InitAttributes(self):
        self.filename = None
        self.vdim = array([0,0,0])
        self.vsize = array([1.0,1.0,1.0])
        self.vunit = 0
        self.nchan = 0
        self.clist = np.zeros(8, dtype=np.uint8)
        self.type = 0
        self.xdim = 0
        self.ydim = 0
        self.zdim = 0
        self.data = None

    def CopyAttributes(self, vol):
        self.CopyMetaInfo(vol)
        self.type = vol.type
        self.vdim = vol.vdim

    def CopyMetaInfo(self, vol):
        self.vsize = vol.vsize
        self.vunit = vol.vunit
        self.nchan = vol.nchan
        self.clist = vol.clist

    def SetVoxelSize(self, vs, vu=None):
        if isinstance(vs, (tuple,list,ndarray)) and len(vs) == 3:
            self.vsize = array(vs, dtype=np.float64)
        if isinstance(vu, int) and vu >= 0:
            self.vunit = vu
        elif isinstance(vu, str):
            try:
                self.vunit = Volume.VOXEL_UNITS[vu]
            except:
                pass

    def Clone(self, data=None, *args):
        d = None
        if isinstance(data,ndarray):
            d = data if len(data.shape) >= 3 else ndarray(data,dtype=self.dtype)
        elif data == 0:
            d = np.zeros_like(self.data)
        elif data == []:
            d = np.empty_like(self.data)
        elif isinstance(data, (tuple,list)):
            d = np.ndarray(data, dtype=self.dtype)
        elif data is None:
            d = self.data.copy()
        v = self.__class__(d, args if bool(args) == True else None)
        v.CopyMetaInfo(self)
        return v

#
#   Data accessing methods
#
    def GetValue(self, x, y, z):
        if 0 <= x < self.xdim and 0 <= y < self.ydim and 0 <= z < self.zdim:
            return self.data[int32(z),int32(y),int32(x)]
        return 0

    def Nonzero(self):
        if self.data is None:
            return []
        return self.data.ravel().nonzero()[0]

    def GetMinMax(self):
        return self.data.min(), self.data.max()

    @staticmethod
    def FFTSize(src, p2=False):
        w, h, d = 0, 0, 0
        if isinstance(src, Volume):
            w, h, d = src.vdim
        elif isinstance(src, np.ndarray) and len(src) == 3:
            w, h, d = src
        else:
            return array([h, h, d])
        return array([CalcFFTSize(w,p2=p2), CalcFFTSize(h,p2=p2), CalcFFTSize(d,p2=p2)])

    def GetFFTSize(self,p2=False):
        return Volume.FFTSize(self,p2=p2)

    def ClipDimensions(self, dim):
        return dim.clip(self.vdim)

#
#   Type converting methods
#
    def AsType(self, dtype):
        if dtype in (float, np.float32, np.float64):
            v = self.data.astype(dtype)
        elif self.data.dtype != dtype:
            #m, x = self.data.min(), self.data.max()
            #d = self.data
            #t = Volume.DTYPE_VMAX[dtype]
            v = nx.evaluate('(d - m) * t / (x - m)',
                            local_dict={
                                'd': self.data,
                                'm': self.data.min(),
                                'x': self.data.max(),
                                't': Volume.DTYPE_VMAX[dtype]
                            }).astype(dtype)
            #v = dtype((self.data - m) * Volume.DTYPE_VMAX[dtype] / (x - m))
        else:
            v = self.data
        return self.Clone(v)

    def AsByte(self):
        return self.AsType(np.uint8)

    def ByteScale(self):
        vm = self.data.min()
        vx = self.data.max() - vm
        self.data = uint8((self.data - vm) * 255.0 / vx)

    def Normalize(self, inplace=False):
        m, x = self.data.min(), self.data.max()
        if self.dtype in (float,np.float32,np.float64):
            x = x - m
        else:
            x = float(x - m)
        r = (self.data - m) / x
        if inplace:
            self.data = r
            return self
        return self.Clone(r)

    def Binarize(self, threshold=0, inplace=False):
        if inplace:
            self.data[self.data > threshold] = 1
            return self
        return self.Clone(uint8(self.data > threshold))

#
#   I/O methods
#
    def Save(self, filename=None):
        try:
            if isinstance(filename,str):
                self.data.tofile(filename)
                self.filename = filename
            elif isinstance(self.filename,str):
                self.data.tofile(self.filename)
        except TypeError:
            pass

#
#   Volume manipulation methods
#
    def Pad(self, psize):
        if isinstance(psize, (tuple,list)):
            psize = array(psize)
        bp = psize // 2
        ep = bp + self.vdim
        vb = np.zeros((self.vdim + psize)[::-1], dtype=self.dtype)
        vb[bp[2]:ep[2],bp[1]:ep[1],bp[0]:ep[0]] = self.data
        return self.Clone(vb)

    def PadTo(self, size):
        return self.Pad(size - self.vdim)

    def Rip(self, rsize):
        sp = (self.vdim - rsize) // 2
        ep = sp + rsize
        rv = self.data[sp[2]:ep[2],sp[1]:ep[1],sp[0]:ep[0]]
        return self.Clone(rv.ravel().reshape(rv.shape))

    def GetFFTVolume(self,p2=False):
        return self.PadTo(self.GetFFTSize(p2=p2))

    def TrimZero(self):
        n = self.Nonzero()
        if len(n) == 0:
            return Volume(self)
        w = self.xdim
        p = w * self.ydim
        z = n // p
        n = n %  p
        y = n // w
        x = n %  w
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        zmin, zmax = z.min(), z.max()
        xdim, ydim, zdim = zmax-zmin+1, ymax-ymin+1, xmax-xmin+1
        data = self.data[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1].copy()
        return self.Clone(data)

    def Mirror(self, ext=None, fftsize=False):
        vp = VolProcess()
        rv = vp.Mirror(self, ext, fftsize)
        vp.Release()
        return rv

    def Extend(self, ext):
        vdim = self.vdim + ext
        data = np.zeros(vdim[::-1], dtype=self.data.dtype)
        data[0:self.zdim,0:self.ydim,0:self.xdim] = self.data
        return self.Clone(data)

    def Crop(self, p, size, out=None, dataonly=False):
        e = p + size
        if isinstance(out, (ndarray,Volume)):
            out[...] = self.data[p[2]:e[2],p[1]:e[1],p[0]:e[0]]
            return out
        v = self.data[p[2]:e[2],p[1]:e[1],p[0]:e[0]].copy()
        if dataonly:
            return v
        return self.Clone(v)

    def Put(self, sp, vol):
        ep = sp + vol.vdim
        self.data[sp[2]:ep[2],sp[1]:ep[1],sp[0]:ep[0]] = vol.data

    def Rescale(self, scales, mode='linear', rsize=None):
        vp = VolProcess()
        rv = vp.Rescale(self, scales, mode, rsize)
        vp.Release()
        return rv
        
    def ZMeanProfile(self, sf=5):
        def Smooth(data, width, mode='reflect'):
            w = width // 2
            d = np.pad(data, w, mode=mode)
            r = np.empty_like(data)
            for i in np.arange(len(data))+w:
                r[i-w] = d[i-w:i+w+1].mean()
            return r
        zm = []
        print 'Measuring...'
        for i in range(self.zdim):
            s = self.data[i,:,:]
            s = s[np.where(s > 0)]
            if len(s) > 0:
                zm.append(s.mean())
            else:
                zm.append(0.0)
        zm = Smooth(np.array(zm), sf)
        return zm / zm.max()

    def Morph(self, opertator, kernel, gray=False):
        kernel = kernel.AsByte().TrimZero().Binarize()
        worker = VolMorphology.Create()
        fmorph = worker.__getattribute__(opertator)
        return self.Clone(fmorph(self.data, kernel.data, Gray=gray))

    def Promote(self, clz):
        if not issubclass(clz, Volume):
            raise TypeError('%s must be a subclass of %s' % (repr(clz), repr(Volume)))
        self.__class__ = clz
        if self.filename is not None:
            self.filename = XFile.NewExt(self.filename, clz.DEFAULT_EXTENSION)

    @classmethod
    def Ones(clz, dim, dtype=np.uint8):
        return clz.Solid(dim, 1, dtype)

    @classmethod
    def Zeros(clz, dim, dtype=np.uint8):
        return clz.Solid(dim, 0, dtype)

    @classmethod
    def Solid(clz, dim, value, dtype=np.uint8):
        if not isinstance(dim, (list,tuple,np.ndarray)) or len(dim) != 3:
            raise TypeError('Volume dimensions must be a list, tuple or array of 3 numbers.')
        if value == 0:
            v = np.zeros(dim[::-1], dtype=dtype)
        elif value == 1:
            v = np.ones(dim[::-1], dtype=dtype)
        else:
            v = np.ndarray(dim[::-1], dtype=dtype)
            v[...] = value
        return clz(v)

    @classmethod
    def Hanning(clz, dim, dtype=float):
        v = np.ndarray(dim[::-1], dtype=float)
        w = hanning(dim[0])
        h = hanning(dim[1])
        d = hanning(dim[2])
        p = matrix(h).T * matrix(w)
        for z in range(dim[2]):
            v[z,:,:] = p * d[z]
        if dtype is not float:
            return clz(v).AsType(dtype)
        return clz(v)

    @classmethod
    def Gaussian(clz, sigmas, dtype=float):
        dim = 2 * (ceil(array(sigmas) * 3).astype(int32) | 1) + 1
        v = np.ndarray(dim[::-1], dtype=float)
        w = WinFx.Gaussian(sigmas[0], dim[0])
        h = WinFx.Gaussian(sigmas[1], dim[1])
        d = WinFx.Gaussian(sigmas[2], dim[2])
        p = matrix(h).T * matrix(w)
        for z in range(dim[2]):
            v[z,:,:] = p * d[z]
        if dtype is not float:
            return clz(v).AsType(dtype)
        return clz(v)

    @classmethod
    def DistanceToCenter(clz, dim, nosqrt=False, center=True):
        dv = clz(np.zeros(dim[::-1], dtype=float))
        vp = VolProcess()
        vp.CenterDistance(dv, nosqrt, center)
        vp.Release()
        return dv.data

    @classmethod
    def Butterworth(clz, dim, D0=10, degree=1.0, highpass=False, center=True, dtype=float):
        D = clz.DistanceToCenter(dim, center=True) / D0
        if highpass:
            z = np.where(D == 0)
            n = np.where(D != 0)
            D[n] = np.power(1.0 / D[n], 2*degree)
            for zt,yt,xt in zip(*z):
                D[zt,yt,xt] = D[zt-1:zt+1,yt-1:yt+1,xt-1:xt+1].sum() / 26.0
        else:
            D = np.power(D, 2*degree)
        H = nx.evaluate('1.0 / (1 + D)')
        if dtype is not float:
            return clz(H).AsType(dtype)
        return clz(H)

    @classmethod
    def Homomorphic(clz, dim, D0=10, GH=2.0, GL=0.5, C=0.5, center=True, dtype=float):
        D = clz.DistanceToCenter(dim, nosqrt=True, center=center)
        H = nx.evaluate('(GH - GL) * (1 - exp(-(C * D / (D0**2.0)))) + GL')
        if dtype is not float:
            return clz(H).AsType(dtype)
        return clz(H)
    @classmethod
    def HomomorphicCPU(clz, dim, D0=10, GH=2.0, GL=0.5, C=0.5, center=True, dtype=float):
        dv = clz(np.zeros(dim[::-1], dtype=float))
        vp = VolProcess()
        vp.Homomorphic(dv, D0, GH, GL, C, nosqrt=True, center=center)
        vp.Release()
        if dtype is not float:
            return clz(dv).AsType(dtype)
        return clz(dv)

    @classmethod
    def Sphere(clz, r):
        return clz.Elipsoid(r,r,r)

    @classmethod
    def Elipsoid(clz, rx, ry, rz):
        def Quadratic(r):
            d = r * 2.0
            d = np.floor(d) + (d - np.floor(d) != 0)
            r = d / 2
            if d % 2 == 0:
                x = np.arange(r) + 0.5
                x = np.concatenate((x[::-1], x)) ** 2
            else:
                x = np.arange(r)
                x = np.concatenate((x[1:][::-1], x)) ** 2
            return x / (r ** 2), d
        def Create(x, y, z, dx, dy, dz):
            p = np.ndarray((dy, dx))
            for i in np.arange(dy):
                p[i,:] = x + y[i]
            v = np.ndarray((dz, dy, dx))
            for i in np.arange(dz):
                v[i,:,:] = p + z[i]
            return (v <= 1.0) * 255

        x, dx = Quadratic(rx)
        y, dy = Quadratic(ry)
        z, dz = Quadratic(rz)
        v = Create(x, y, z, dx, dy, dz)
        return clz(v.astype(uint8))

    def ApplyMask(self, mfx):
        return eval('self.%s()' % mfx)

    def ExponentialMask(self):
        w, h, d = self.vdim
        wx = WinFx.Exponential(w)
        wy = WinFx.Exponential(h)
        wz = WinFx.Exponential(d)
        xy = matrix(wy).T * matrix(wx)
        for z in range(d):
            self.data[z,:,:] *= xy * wz[z]
        return self

    def GaussianMask(self):
        w, h, d = self.vdim
        wx = WinFx.Gaussian(CalcGaussianSigma(w), w)
        wy = WinFx.Gaussian(CalcGaussianSigma(h), h)
        wz = WinFx.Gaussian(CalcGaussianSigma(d), d)
        xy = matrix(wy).T * matrix(wx)
        for z in range(d):
            self.data[z,:,:] *= xy * wz[z]
        return self

#
#   Filter methods
#
    @VolumeController
    class Filter(object):
        @classmethod
        def Kernel(clz, k):
            if isinstance(k, Volume):
                return k
            if isinstance(k, int):
                return k    #Volume.Ones([k,k,k])
            if isinstance(k, (list,tuple,np.ndarray)) and len(k) == 3:
                return Volume.Ones(k)
            raise TypeError('kernel must be an instance of '+repr(Volume))
        @classmethod
        def Median(clz, kernel, cdf=False):
            vf = VolFilters()
            if cdf:
                rv = vf.MedianCDF(clz.vol, clz.Kernel(kernel))
            else:
                rv = vf.Median(clz.vol, clz.Kernel(kernel))
            vf.Release()
            return rv
        @classmethod
        def Mean(clz, kernel):
            vf = VolFilters()
            rv = vf.Mean(clz.vol, clz.Kernel(kernel))
            vf.Release()
            return rv
        @classmethod
        def Sigma(clz, kernel, params=[2.0, 0.2]):
            sf = params; sf.append(Volume.DTYPE_VMAX[clz.vol.dtype.type])
            vf = VolFilters()
            rv = vf.Sigma(clz.vol, clz.Kernel(kernel), sf)
            vf.Release()
            return rv
        @classmethod
        def Anisotropic(clz, K=180.0, DT=1.0/14, N=10):
            params = np.array([K, DT, N], dtype=float)
            vf = VolFilters()
            rv = vf.Anisotropic(clz.vol, params)
            vf.Release()
            return rv.AsType(clz.vol.dtype.type)
        @classmethod
        def Laplace(clz, Lambda=0.05, N=5):
            params = np.array([0, Lambda, N], dtype=float)
            vf = VolFilters()
            rv = vf.Laplace(clz.vol, params)
            vf.Release()
            return rv.AsType(clz.vol.dtype.type)
        @classmethod
        def LocalNormalize(clz, sigm=[2.0]*3, sigv=[20.]*3):
            mv = clz.Convol(Volume.Gaussian(sigm))
            rv = np.abs(np.subtract(clz.vol.data, mv.data))
            mv.data = np.power(rv, 2.0)
            mv = mv.Filter.Convol(Volume.Gaussian(sigv))
            rv = np.divide(rv, np.sqrt(mv.data))
            rv[np.isinf(rv)] = 0
            return clz.vol.Clone(rv)
        @classmethod
        def Gaussian(clz, sigmas, fast=True):
            kernel = Volume.Gaussian(sigmas)
            return clz.Convol(kernel, fast=fast)
        @classmethod
        def Hanning(clz, dim=None):
            if not isinstance(dim,(list,tuple)):
                dim = clz.vol.vdim
            kernel = Volume.Hanning(dim)
            return clz.Convol(kernel)
        @classmethod
        def Convol(clz, kernel, pad=None, fast=True, same_dtype=False):
            if pad is None:
                pad = clz.vol.vdim // 8;
            cv = clz.vol.Mirror(pad, fftsize=True)
            kv = clz.Kernel(kernel)
            kv = kv.Pad(cv.vdim - kv.vdim)
            if fast:
                kv = kv.AsType(np.float32)
                cv = cv.AsType(np.float32)
            else:
                kv = kv.AsType(np.float64)
                cv = cv.AsType(np.float64)
            vf = VolFilters()
            cv = clz.vol.Clone(vf.Convol(cv, kv))
            vf.Release()

            if same_dtype:
                return cv.Rip(clz.vol.vdim).AsType(clz.vol.data.dtype)
            return cv.Rip(clz.vol.vdim)

        @classmethod
        def SpatialConvol(clz, kernel, fast=True, same_dtype=False):
            kv = clz.Kernel(kernel)
            if fast:
                kv = kv.AsType(np.float32)
                #cv = clz.vol.AsType(np.float32)
            else:
                kv = kv.AsType(np.float64)
                #cv = clz.vol.AsType(np.float64)
            cv = clz.vol.Clone()
            print "Src dtype = ", cv.dtype
            vf = VolFilters()
            cv = vf.SpatialConvol(cv, kv)
            vf.Release()

            if same_dtype:
                return cv.AsType(clz.vol.data.dtype)
            return cv

        @classmethod
        def Kalman(clz, direction, gain=0.8, nvar=0.05):
            def GetPlane(vol, direction, n, out=None):
                o = None
                if direction == 0:
                    o = vol[:,:,n].astype(np.float64)
                elif direction == 1:
                    o = vol[:,n,:].astype(np.float64)
                else:
                    o = vol[n,:,:].astype(np.float64)
                if out is None:
                    return o
                out[...] = o

            def SetPlane(vol, direction, n, value):
                if direction == 0:
                    vol[:,:,n] = value
                elif direction == 1:
                    vol[:,n,:] = value
                else:
                    vol[n,:,:] = value

            try:
                direction = {'x':0,'y':1,'z':2}[direction]
            except:
                print "Direction must be in ['x','y','z']"
                return None

            vol = clz.vol.Clone()
            P = GetPlane(vol, direction, 0)
            V = np.ndarray(P.shape); V[...] = nvar
            N = V.copy()
            K = np.empty_like(N)
            P = np.empty_like(N)
            O = np.empty_like(N)
            for n in xrange(vol.vdim[direction] - 1):
                GetPlane(vol, direction, n+1, out=O)
                nx.evaluate('N / (N + V)', out=K)
                nx.evaluate('gain * P + (1.0 - gain) * O + K * (O - P)', out=P)
                nx.evaluate('N * (1.0 - K)', out=N)
                SetPlane(vol, direction, n+1, P)
            return vol

        @classmethod
        def Homomorphic_NP(clz, D0=10, GH=1.5, GL=0.5, C=0.5, fast=True):
            cv = nx.evaluate('log(d + 1.0)', local_dict={'d': clz.vol.data})
            cv = clz.vol.Clone(data=cv)
            cv = cv.Mirror(clz.vol.vdim // 8, fftsize=True)
            kv = Volume.Homomorphic(cv.vdim, D0, GH, GL, C, center=False)
            if fast:
                kv = kv.AsType(np.float32)
                cv = cv.AsType(np.float32)
            else:
                kv = kv.AsType(np.float64)
                cv = cv.AsType(np.float64)
            cv.data = np.real(np.fft.ifftn(np.fft.fftn(cv.data) * kv.data))
            cv.data = nx.evaluate('exp(d) - 1.0', local_dict={'d': cv.data}).clip(0)
            return cv.Rip(clz.vol.vdim)
        @classmethod
        def Homomorphic(clz, D0=10, GH=1.5, GL=0.5, C=0.5, fast=True):
            cv = nx.evaluate('log(d + 1.0)', local_dict={'d': clz.vol.data})
            cv = clz.vol.Clone(data=cv)
            cv = cv.Mirror(clz.vol.vdim // 8, fftsize=True)
            kv = Volume.Homomorphic(cv.vdim, D0, GH, GL, C, center=False)
            kv = Volume(kv[:,:,0:kv.vdim[0]/2+1].copy())
            if fast:
                kv = kv.AsType(np.float32)
                cv = cv.AsType(np.float32)
            else:
                kv = kv.AsType(np.float64)
                cv = cv.AsType(np.float64)
            vf = VolFilters()
            vf.FrequencyMask(cv, kv)
            vf.Release()
            cv.data = nx.evaluate('exp(d) - 1.0', local_dict={'d': cv.data}).clip(0)
            return cv.Rip(clz.vol.vdim)
