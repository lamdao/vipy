# -*- coding: utf-8 -*-
"""
Created on Mon Dec 02 16:28:05 2013

@author: Lam H. Dao  <lam(dot)dao(at)nih(dot)gov>
                     <daohailam(at)yahoo(dot)com>
"""

import numpy as np

def CurveFit(x, y, degree=3):
    m = degree+1
    k = np.argsort(x)
    c,r,_,_,_ = np.polyfit(x[k], y[k], degree, full=True)
    e = np.sqrt(r[0] / (len(x)-m))
    p = np.poly1d(c)
    x = np.unique(x[k])
    f = p(x)
    return x, f, e

def BlobFit(vol, degree=3, full=False):
    """This function uses polynomial fitting to estimate a 3D elongated
       binary structure and returns coordinate of its center line

       Parameters:
       - vol     - MIV volume stores binary structure
       - degree  - Polynomial degree to fit
       - full    - Include coordinate at both ends of the structure
    """
    w = vol.xdim
    p = vol.xdim * vol.ydim
    n = vol.Nonzero()
    z = n // p; n = n % p
    y = n // w
    x = n %  w

    rx = x.ptp()    #rx = max(x) - min(x)
    ry = y.ptp()    #ry = max(y) - min(y)
    rz = z.ptp()    #rz = max(z) - min(z)

    if rx > ry:
        if rx > rz:
            _, y, e1 = CurveFit(x, y, degree)
            x, z, e2 = CurveFit(x, z, degree)
        else:
            _, y, e1 = CurveFit(z, y, degree)
            z, x, e2 = CurveFit(z, x, degree)
    else:
        if ry > rz:
            _, x, e1 = CurveFit(y, x, degree)
            y, z, e2 = CurveFit(y, z, degree)
        else:
            _, y, e1 = CurveFit(z, y, degree)
            z, x, e2 = CurveFit(z, x, degree)

    e = int(np.ceil(e1 if e1 > e2 else e2))
    if full:
        return x, y, z, e
    return x[e:-e-1], y[e:-e-1], z[e:-e-1]

