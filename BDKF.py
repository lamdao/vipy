# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04 15:17:42 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

from MIVFile import MIV
from ExtLib import ExternalLib
from FileRoutines import XFile

from multiprocessing import cpu_count
from ctypes import c_int32, byref
from numpy import array
import numpy as np
import time

class BDKF(ExternalLib):
    """ 3D Blind Deconvolution with Kalman Filter
    This is a wrapper class for BDKF.{dll,so,dylib} written in C/C++
    """
    ncpu = c_int32(cpu_count())
    wpar = np.array([0.0]*7)

    def __init__(self):
        ExternalLib.Init(self, 'BDKF')

    def SetWorkingSize(self, dim):
        self.AddParameters(dim, byref(BDKF.ncpu))
        self.Execute(self.lib.vol_bdkf_init)

    def SetParameters(self, vstop, kstop, rlambda, dmax, cmax, niters, verbose=0):
        self.params = np.array([vstop, kstop, rlambda, dmax, cmax, niters, verbose], dtype=np.float64)

    def SetFilterParameters(self, noisevar=0.05, gainvalue=0.5):
        self.vflter = array([noisevar, gainvalue], dtype=np.float64)
        self.AddParameters(self.vflter)
        self.Execute(self.lib.vol_bdkf_pset)

    dump_params = False
    def ShowParameters(self, params):
        if not BDKF.dump_params:
            return
        print """Deconvolving with parameters:
        - vstop: %.1e
        - kstop: %.1e
        - niter: %d
        - vfltr: %.1e %.2f
        """ % (params[0], params[1], params[5], self.vflter[0], self.vflter[1])
        print params[2:5]

    def Invoke(self, volume, kernel, output):
        self.wdim = volume.cdim
        BDKF.wpar = self.params.copy()
        self.AddParameters(self.wdim, volume.data, kernel.data, output.data, BDKF.wpar)
        self.ShowParameters(BDKF.wpar)
        self.Execute(self.lib.vol_bdkf_exec)

#----------------------------------------------------------------------------
# Adaptive spatial BDKF
#
# Apply BDKF using block-by-block deconvolution
def SpatialApply(vf, kf, vstop=0.02, kstop=0.05, rlambda=1E-4, niters=100,
                 nvar=0.1, gain=0.5, fast=True, out=None,
                 savedf=False, verbose=0, bs=64):

    t0 = time.time()
    wtype = np.float32 if fast else np.float64
    volume = MIV(vf, dtype=wtype)
    kernel = MIV(kf, dtype=wtype)

    d = volume.vdim
    s = np.array([bs]*3)
    u = (d // s + ((d % s) > 0)) * s
    p = u - d
    e = s
    v = volume.Extend(p).Mirror(e); del volume
    n = v.vdim - s
    o = e / 2

    nc = 0
    ws = (s + e).clip(kernel.vdim)

    rv = v.Clone(data=[])
    if savedf:
        kv = v.Clone(data=[])

    k0 = kernel.Normalize().PadTo(ws); del kernel
    bo = k0.Clone(data=[])
    bv = None

    dm = vstop / 10
    cm = 100

    dcv = BDKF()
    dcv.SetWorkingSize(k0.cdim)
    dcv.SetParameters(vstop, kstop, rlambda, dm, cm, niters, verbose)
    dcv.SetFilterParameters(nvar, gain)
    for z in np.arange(o[2],n[2]+1,s[2]):
        for y in np.arange(o[1],n[1]+1,s[1]):
            for x in np.arange(o[0],n[0]+1,s[0]):
                nc += 1
                t1 = time.time()
                if verbose > 0:
                    print ',-- [ %03d (%03d,%03d,%03d) ] --------' % (nc,x,y,z)
                ks = k0.Clone()
                bv = v.Crop([x,y,z] - o, ws, bv)
                dcv.Invoke(bv, ks, bo)
                t1 = time.time() - t1
                if verbose > 0:
                    print '| #%d: %.1e %.1e' % (BDKF.wpar[5], BDKF.wpar[0], BDKF.wpar[1])
                    print '`- Time: %.1fs' % (t1)

                rv.Put([x,y,z], bo.Rip(s))
                if savedf:
                    kv.Put([x,y,z], ks.Rip(s))

    if savedf:
        kv.Save()
    dcv.Release()

    rv.Rip(d+p).Crop([0,0,0], d).AsByte().Save(filename=out)
    print 'Total time: %.1fs' % (time.time()-t0)
#----------------------------------------------------------------------------
# Apply BDKF globally (full volume)
def GlobalApply(vfn, kfn, vstop=0.02, kstop=0.05, rlambda=1E-4, niters=100,
                nvar=0.1, gain=0.5, prefilter=None, out=None, omf=False, fast=True,
                vext=True, savedf=False, verbose=0, p2fft=False):
    wtype = np.float32 if fast else np.float64
    volume = MIV(vfn, dtype=wtype)
    kernel = MIV(kfn, dtype=wtype)

    v = None
    if vext is 'fft':
        v = volume.GetFFTVolume(p2=True)
    elif vext is True:
        v = volume.Mirror(volume.vdim//8).GetFFTVolume()
    elif vext is 0:
        v = volume.Pad(volume.vdim//8).GetFFTVolume()
    else:
        pdim = MIV.FFTSize(volume.vdim, p2=p2fft) - volume.vdim
        v = volume.Mirror(pdim)
        #v = volume.GetFFTVolume()
    s = v.vdim.clip(kernel.vdim)
    print 'PadTo: ', s
    k = kernel.PadTo(s)
    v = v.PadTo(s)
    if isinstance(prefilter,str):
        try:
            v = v.ApplyMask(prefilter)
            k = k.ApplyMask(prefilter)
        except:
            print 'Masking method %s not found' % prefilter
            return
    r = v.Clone(data=[])
    dm = vstop / 10
    cm = int((5 + np.log10(vstop)) * 200)

    dcv = BDKF()
    dcv.SetWorkingSize(k.cdim)
    dcv.SetParameters(vstop, kstop, rlambda, dm, cm, niters, verbose)
    dcv.SetFilterParameters(nvar, gain)
    dcv.Invoke(v, k, r)
    dcv.Release()
    r = r.Rip(volume.vdim)
    if out is None:
        return r
    r.Save(filename=out)
    if isinstance(omf,int) and omf > 2:
        r.Filter.Median(omf).Save(XFile.NewExt(out, 'mf%d.miv' % omf))
    if savedf is True:
        k.Rip(kernel.vdim).Save(XFile.NewExt(out, 'kdf.miv'))

