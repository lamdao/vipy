# -*- coding: utf-8 -*-
"""
Created on Mon Jun 09 13:53:21 2014

@author: Lam H. Dao <lam(dot)dao(at)nih(dot)gov>
                    <daohailam(at)yahoo(dot)com>
"""
import numpy as np

from DFastICA import DFastICA
from SimpleMIV import MIV_Load

def Test_DFastICA():
    c1,_ = MIV_Load(r'data/c1.miv')
    c2,_ = MIV_Load(r'data/c2.miv')
    c3,d = MIV_Load(r'data/c3.miv')
    print "Image size =", d[:2]
    Dx = np.vstack((c1.ravel(), c2.ravel(), c3.ravel()))
    fica = DFastICA()

# Estimate mixing matrix
#    Mx = fica.Run(Dx)
#    print Mx

# Unmix with supplied mixing matrix
#    Mx = np.array([
#            [ 0.71261023, 0.02176309, 0.03955992],
#            [ 0.42966614, 0.40058698, 0.17760640],
#            [ 0.03191832, 0.04979355, 0.40215257]])
#    Mx, Ux = fica.Run(Dx, umix=Mx)

# Estimate and unmix
    Mx, Ux = fica.Run(Dx, umix=True)
    Ux = Ux.reshape((d[1],d[0],3))
    print "Mx ="
    print Mx
    print "Unmixed data ="
    print type(Ux), Ux.shape

def Test_Stack_DFastICA():
    c1,_ = MIV_Load(r'data/s1.miv')
    c2,_ = MIV_Load(r'data/s2.miv')
    c3,d = MIV_Load(r'data/s3.miv')
    print "Stack size =", d
    Dx = np.vstack((c1.ravel(), c2.ravel(), c3.ravel()))
    fica = DFastICA()

    Mx, Ux = fica.Run(Dx, umix=True)
    Ux = Ux.reshape((d[2],d[1],d[0],3))
    print "Mx ="
    print Mx
    print "Unmixed data ="
    print type(Ux), Ux.shape

if __name__ == "__main__":
    Test_DFastICA()
