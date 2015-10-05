# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:17:34 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import os
import time
import threading
import numpy as np
import numexpr as nx
from scipy.ndimage import measurements as ms
from VolUtils import Volume
from MIVFile import MIV
from FileRoutines import XFile
#------------------------------------------------------------------------------
def EliminateSmallBlobs(vol, d=13):
    print 'Removing small blobs...'
    t = np.round(np.pi * (d / 2.)**3)
    p, count = ms.label(vol)
    p = p.ravel()
    h = np.histogram(p, bins=count, range=[0,count])[0]; h[0] = 0
    k = (np.where(h > t))[0]
    print ' - Total %d blobs, where %d have size > sphere(radius=%.1f)' % \
            (p.max(), len(k), d/2.)
    v = len(h) + 1
    a = (np.where(p > 0))[0]
    s = p[a]
    for r in range(len(k)):
        p[a[s == k[r]]] = v
    return np.uint8(p == v).reshape(vol.shape)

#------------------------------------------------------------------------------
def SimpleThreshold(im, sf=2.0):
    idx = im.ravel().nonzero()[0]
    if len(idx) <= 0:
        return 1
    vv = im.ravel()[idx]
    return np.round(vv.mean() + sf * vv.std())

#------------------------------------------------------------------------------
def GetThresholdSurface(c, t, mt):
    st = mt if t < mt else t
    return st * c / c.max()

#------------------------------------------------------------------------------
def BinXY(v, sv, mt, sf, res=None):
    b = np.empty_like(v.data, dtype=np.uint8)
    sf = 2.75 - np.arange(v.zdim, dtype=float) / v.zdim
    for z in range(v.zdim):
        s = v.data[z,:,:]
        t = SimpleThreshold(s, sf[z])
        t = GetThresholdSurface(sv.data[z,:,:], t, mt)
        b[z,:,:] = np.uint8((s - t) > 0.5)
    if res == None:
        return b
    res[2] = b

#------------------------------------------------------------------------------
def BinXZ(v, sv, mt, sf, res=None):
    b = np.empty_like(v.data, dtype=np.uint8)
    for y in range(v.ydim):
        s = v.data[:,y,:]
        t = SimpleThreshold(s, sf)
        t = GetThresholdSurface(sv.data[:,y,:], t, mt)
        b[:,y,:] = np.uint8((s - t) > 0.5)
    if res == None:
        return b
    res[0] = b

#------------------------------------------------------------------------------
def BinYZ(v, sv, mt, sf, res=None):
    b = np.empty_like(v.data, dtype=np.uint8)
    for x in range(v.xdim):
        s = v.data[:,:,x]
        t = SimpleThreshold(s, sf)
        t = GetThresholdSurface(sv.data[:,:,x], t, mt)
        b[:,:,x] = np.uint8((s - t) > 0.5)
    if res == None:
        return b
    res[1] = b

#------------------------------------------------------------------------------
def MTBinXY(tid, v, sv, mt, sf, b, nt):
    sf = 2.75 - np.arange(v.zdim, dtype=float) / v.zdim
    z = tid
    while (z < v.zdim):
        s = v.data[z,:,:]
        t = SimpleThreshold(s, sf[z])
        t = GetThresholdSurface(sv.data[z,:,:], t, mt)
        b[z,:,:] = np.uint8((s - t) > 0.5)
        z += nt
#------------------------------------------------------------------------------
def MTBinXZ(tid, v, sv, mt, sf, b, nt):
    y = tid
    while y < v.ydim:
        s = v.data[:,y,:]
        t = SimpleThreshold(s, sf)
        t = GetThresholdSurface(sv.data[:,y,:], t, mt)
        b[:,y,:] = np.uint8((s - t) > 0.5)
        y += nt
#------------------------------------------------------------------------------
def MTBinYZ(tid, v, sv, mt, sf, b, nt):
    x = tid
    while x < v.xdim:
        s = v.data[:,:,x]
        t = SimpleThreshold(s, sf)
        t = GetThresholdSurface(sv.data[:,:,x], t, mt)
        b[:,:,x] = np.uint8((s - t) > 0.5)
        x += nt
#------------------------------------------------------------------------------
def PSeg(fx, v, sv, mt, sf):
    ts = []
    nt = 12
    b = np.empty_like(v.data, dtype=np.uint8)
    for tid in range(nt):
        t = threading.Thread(target=fx, args=(tid, v, sv, mt, sf, b, nt))
        ts.append(t)
        t.start()
    for t in ts:
        t.join()
    return b
#------------------------------------------------------------------------------
def DynaBinarize(v, sv, mk, mt=None, cleanup=True, **kw):
    mt = kw['mt'] if kw.has_key('mt') else np.uint8(0.05 * v.data.max())
    sf = float(kw['sf']) if kw.has_key('sf') else 2.0

#    b = BinXY(v, sv, mt, sf) & BinXZ(v, sv, mt, sf) & BinYZ(v, sv, mt, sf)

    b = PSeg(MTBinXY, v, sv, mt, sf)
    b = b & PSeg(MTBinXZ, v, sv, mt, sf)
    b = b & PSeg(MTBinYZ, v, sv, mt, sf)
    if cleanup:
        return v.Clone(EliminateSmallBlobs(b, **kw) * 255)
    return v.Clone(b * 255)

#------------------------------------------------------------------------------
# Filter Mode:
#
#   'ln' - LocalNormalize
#           mk=[2.]*3, sk=[20.]*3, pfk=[1.5]*3
#
#   'mo' - GaussianFilter+MorphologyClose
#           mk=[.75]*3, sk=[2.]*3, pfk=None
#
#   'gf' - GaussianFilter
#           sk=[2.]*3
#
#------------------------------------------------------------------------------
def MIV_Seg(fn=None, mk=[1.0]*3, sk=[20.0]*3, pfk=[1.5]*3,
            fltmode='ln', pad=None, out=None, **kw_args):

    if not isinstance(fn,str):
        fn = raw_input('Input MIV file: ')
        if fn == '':
            return
    if isinstance(out, bool) and out == True:
        out = None
    elif out is MIV:
        pass
    elif not isinstance(out,str):
        out = XFile.NewExt(fn, 'bin.miv')

    t0 = time.time()

    print 'Starting QuickSeg("%s", mk=%s, sk=%s)' % (fn, str(mk), str(sk))
    print '- Loading...'
    v = MIV(fn)
    if isinstance(pad, np.ndarray) and len(pad.shape) == 1 and pad.size == 3:
        v = v.Pad(pad)

    if fltmode == 'ln':
        print '- Local Normalizing...'
        p = v.Filter.LocalNormalize(sigm=mk, sigv=sk)
        print '- GF Smoothing...'
        p = p.GetFFTVolume()
        if pfk != None:
            p = p.Filter.Gaussian(pfk)
    elif fltmode == 'mo':
        print '- Morphing...'
        p = v.Morph('Close', Volume.Gaussian(mk), gray=True)
        print '- GF Smoothing...'
        p = p.GetFFTVolume()
        p = p.Filter.Gaussian(sk)
    elif fltmode == 'gf':
        print '- GF Smoothing...'
        p = v.GetFFTVolume()
        p = p.Filter.Gaussian(sk)
    elif fltmode == 'mf':
        print '- MF Smoothing...'
        p = v.GetFFTVolume()
        k = Volume.Ones(tuple(mk));
        p = p.Filter.Median(k, cdf=True)
    elif fltmode == 'af':
        print '- AF Smoothing...'
        p = v.GetFFTVolume()
        p = p.Filter.Anisotropic(N=mk[0])
    else:
        print '- No Smoothing...'
        p = v.GetFFTVolume()

    print '- Local mean calculating...'
    s = p.Filter.Hanning().Rip(v.vdim)
    p = p.Rip(v.vdim).AsByte()
    print '- Binarizing...'
    v = DynaBinarize(p, s, mk, **kw_args)
    if isinstance(out, str) or out is None:
        v.Save(filename=out)

    print 'Elapse: %.1fs' % (time.time()-t0)
    if out is MIV:
        return v
#------------------------------------------------------------------------------
def SegBulk(sdir, odir, mk=None, sk=None, ft='mf', pattern='*.miv'):
    if not XFile.PathExists(odir):
        os.mkdir(odir)

    if mk is None:
        mk = [3] * 3
    if sk is None:
        sk = [0.5] * 3

    t0 = time.time()
    files = XFile.GetList(sdir, pattern)
    for f in files:
        fn = sdir + '/' + f
        fo = odir + '/' + XFile.NewExt(f, 'bin.miv')
        MIV_Seg(fn, mk=mk, sk=sk, pfk=[.5]*3, fltmode=ft, out=fo)
    print 'Total time: %.1fs' % (time.time()-t0)
#------------------------------------------------------------------------------
if __name__ == '__main__':
    MIV_Seg('D:/Data/vessel.sample.miv')
