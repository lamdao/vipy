# -*- coding: utf-8 -*-
"""
Created on Sun Jun 08 19:13:39 2014

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""
from ExtLib import ExternalLib
from ctypes import c_voidp, c_int32, c_double, POINTER
import numpy as np

class DFastICA(ExternalLib):
    """This is a wrapper class for FastICA.{dll,so,dylib} written in C/C++

    >>> # Get sample data at
    >>> # https://www.dropbox.com/sh/4mgijapnn7hh83a/AAA9INbf2bj1IRU6gXW3AnMAa
    >>> import numpy as np
    >>> import SimpleMIV
    >>> c1,_ = MIV_Load(r'data\c1.miv')
    >>> c2,_ = MIV_Load(r'data\c2.miv')
    >>> c3,d = MIV_Load(r'data\c3.miv')
    >>> print "Image size =", d[:2]
    >>> Dx = np.vstack((c1.ravel(), c2.ravel(), c3.ravel()))
    >>> from DFastICA import DFastICA
    >>> dfica = DFastICA()
    >>> # Estimate mixing matrix
    >>> Mx = dfica.Run(Dx)
    >>> print Mx
    >>> # Unmix with supplied mixing matrix
    >>> Mx = np.array([
    ...              [ 0.71261023, 0.02176309, 0.03955992],
    ...              [ 0.42966614, 0.40058698, 0.17760640],
    ...              [ 0.03191832, 0.04979355, 0.40215257]])
    >>> Mx, Ux = dfica.Run(Dx, umix=Mx)
    >>> # Estimate and unmix
    >>> Mx, Ux = dfica.Run(Dx, umix=True)
    >>> Ux = Ux.reshape((d[1],d[0],3))
    >>> print "Mx ="
    >>> print Mx
    >>> print "Unmixed data ="
    >>> print type(Ux), Ux.shape
    """
    def __init__(self):
        self.Init(self, 'FastICA')
        self.DEstimate = self.lib.Estimate
        self.DEstimate.restype = POINTER(c_double * 64)
        self.DEstimate.argstype = [c_voidp, c_int32, c_int32, c_int32]
        self.DSeparate = self.lib.Unmix
        self.DSeparate.restype = POINTER(c_double * 64)
        self.DSeparate.argstype = [c_voidp, c_voidp, c_int32, c_int32, c_int32]

    def Estimate(self, Dx, nc, ns):
        Rx = self.DEstimate(Dx.ctypes, 2, nc, ns)
        return np.array(Rx.contents[0:nc * nc]).reshape((nc, nc))

    def Separate(self, Mx, Dx, nc, ns):
        Rx = Mx if isinstance(Mx, c_voidp) else Mx.ctypes
        Rx = self.DSeparate(Rx, Dx.ctypes, 2, nc, ns)
        return np.array(Rx.contents[0:nc * nc]).reshape((nc, nc))

    def Run(self, sources, umix=None):
        if not isinstance(sources, np.ndarray):
            print "Sources must be an instance of numpy.ndarray"
            return None if umix is None else None,None
        if len(sources.shape) != 2:
            print "Sources must be a 2D array"
            return None if umix is None else None,None
        Dx = sources.astype(np.uint16)
        ns, nc = sources.shape
        if ns < nc:
            Dx = Dx.T.copy()
            nc, ns = ns, nc
        Mx = c_voidp(None)
        if umix is None:
            umix = False
        elif isinstance(umix, bool):
            pass
        elif not isinstance(umix, np.ndarray):
            umix = False
        elif umix.shape[0] == nc and umix.shape[1] == nc:
            Mx = umix
            umix = True
        else:
            print "Number of components and Mixing matrix are mismatch"
            return None, None
        if umix:
            Rx = self.Separate(Mx, Dx, nc, ns)
            return Rx, Dx
        return self.Estimate(Dx, nc, ns)
