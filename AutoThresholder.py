# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:52:42 2015

Original ImageJ plugin version:
  Autothresholding methods from the Auto_Threshold plugin
  (http://pacific.mpi-cbg.de/wiki/index.php/Auto_Threshold)
  by G.Landini at bham dot ac dot uk.
  
  Latest source:
  http://rsbweb.nih.gov/ij/developer/source/ij/process/AutoThresholder.java.html

Python version:
  Ported by Lam H. Dao <lam(dot)dao(at)nih(dot)gov>
                       <daohailam(at)yahoo(dot)com>
"""
#------------------------------------------------------------------------------
import numpy as np
#------------------------------------------------------------------------------
#
class Methods():
#    Default         = 0
    DefaultIsoData  = 0
    Huang           = 1
    IJIsoData       = 2
    Intermodes      = 3
    IsoData         = 4
    Li              = 5
    MaxEntropy      = 6
    Mean            = 7
    MinError        = 8
    Minimum         = 9
    Moments         = 10
    Otsu            = 11
    Percentile      = 12
    RenyiEntropy    = 13
    Shanbhag        = 14
    Triangle        = 15
    Yen             = 16
#------------------------------------------------------------------------------
#
def Huang(data):
    """
    Implements Huang's fuzzy thresholding method 
    Uses Shannon's entropy function (one can also use Yager's entropy function) 
    Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
    the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51
    """
    # Determine the first & last non-zero bin
    nz = np.where(data != 0)[0]
    first_bin, last_bin = nz[0], nz[-1]

    term = 1.0 / (last_bin-first_bin)
    mu_0 = np.ndarray(256)
    sum_pix = num_pix = 0
    for ih in range(first_bin, 256):
        sum_pix += float(ih) * data[ih]
        num_pix += data[ih]
        # NUM_PIX cannot be zero !
        mu_0[ih] = sum_pix / num_pix

    mu_1 = np.ndarray(256)
    sum_pix = num_pix = 0.0
    for ih in range(last_bin, 0, -1):
        sum_pix += float(ih) * data[ih]
        num_pix += data[ih]
        # NUM_PIX cannot be zero !
        mu_1[ih - 1] = sum_pix / num_pix

    # Determine the threshold that minimizes the fuzzy entropy
    threshold = -1
    min_ent = np.finfo(np.float).max
    for it in range(0, 256):
        ent = 0.0
        for ih in range(0, it+1):
            # Equation (4) in Ref. 1
            mu_x = 1.0 / (1.0 + term * np.abs(ih - mu_0[it]))
            if not ((mu_x  < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Ref. 1 */
                ent += data[ih] * (-mu_x * np.log(mu_x) - \
                       (1.0 - mu_x) * np.log(1.0 - mu_x))

        for ih in range(it + 1, 256):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * np.abs(ih - mu_1[it]))
            if not ((mu_x  < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Ref. 1
                ent += data[ih] * (-mu_x * np.log(mu_x) - \
                       (1.0 - mu_x) * np.log(1.0 - mu_x))
        # No need to divide by NUM_ROWS * NUM_COLS * LOG(2) !
        if ent < min_ent:
            min_ent = ent
            threshold = it
    return threshold
#------------------------------------------------------------------------------
#
def _isBimodal(d):
    n = len(d)
    modes = 0
    for k in range(1,n-1):
        if d[k-1] < d[k] and d[k+1] < d[k]:
            modes = modes + 1
            if modes > 2:
                return False
    return modes == 2
#------------------------------------------------------------------------------
#
def Intermodes(data):
    """
    J. M. S. Prewitt and M. L. Mendelsohn, "The analysis of cell images," in
    Annals of the New York Academy of Sciences, vol. 128, pp. 1035-1053, 1966.
    ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
    Original Matlab code Copyright (C) 2004 Antti Niemisto
    See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
    and the original Matlab code.

    Assumes a bimodal histogram. The histogram needs is smoothed (using a
    running average of size 3, iteratively) until there are only two local maxima.
    j and k
    Threshold t is (j+k)/2.
    Images with histograms having extremely unequal peaks or a broad and
    flat valleys are unsuitable for this method.
    """
    nz = np.where(data > 0)[0]
    minbin, maxbin = nz[0], nz[-1]
    length = (maxbin - minbin) + 1
    hist = data[minbin:maxbin+1].copy()
        
    niter = 0
    while not _isBimodal(hist):
         #smooth with a 3 point running mean filter
        previous = current = 0.0
        hnext = hist[0]
        for i in range(0,length-1):
            previous, current, hnext = current, hnext, hist[i + 1]
            hist[i] = (previous + current + hnext) / 3
        hist[length-1] = (current + hnext) / 3
        niter += 1
        if niter > 10000:
            print("Intermodes Threshold not found after 10000 iterations.")
            return -1

    # The threshold is the mean between the two peaks.
    tt = 0
    for i in range(1, length - 1):
        if (hist[i-1] < hist[i]) and (hist[i+1] < hist[i]):
            tt += i
    return int(tt / 2.0) + minbin
#------------------------------------------------------------------------------
#
def IsoData(data):
    """
    Also called intermeans
    Iterative procedure based on the isodata algorithm [T.W. Ridler, S. Calvard,
    Picture thresholding using an iterative selection method, IEEE Trans.
    System, Man and Cybernetics, SMC-8 (1978) 630-632.]

    The procedure divides the image into objects and background by taking an
    initial threshold, then the averages of the pixels at or below the threshold
    and pixels above are computed.

    The averages of those two values are computed, the threshold is incremented
    and the process is repeated until the threshold is larger than the composite
    average. That is,
      threshold = (average background + average objects)/2

    From: Tim Morris (dtm@ap.co.umist.ac.uk)
    Subject: Re: Thresholding method?
    posted to sci.image.processing on 1996/06/24
    The algorithm implemented in NIH Image sets the threshold as that grey
    value, G, for which the average of the averages of the grey values
    below and above G is equal to G. It does this by initialising G to the
    lowest sensible value and iterating:

    L = the average grey value of pixels with intensities < G
    H = the average grey value of pixels with intensities > G
    is G = (L + H)/2?
        yes => exit
        no  => increment G and repeat
    """
    g = (np.where(data > 0)[0])[0] + 1
    while True:
        l = 0.0
        totl = 0.0
        for i in range(0, g):
             totl = totl + data[i]
             l = l + (data[i] * i)
        h = 0.0
        toth = 0.0
        for i in range(g + 1, 256):
            toth += data[i]
            h += float(data[i]) * i
        if (totl > 0) and (toth > 0):
            l /= totl
            h /= toth
            if g == int(np.round((l + h) / 2.0)):
                break
        g += 1
        if g > 254:
            return -1
    return g
#------------------------------------------------------------------------------
#
def DefaultIsoData(data):
    """
    This is the modified IsoData method used by the "Threshold" widget
    in "Default" mode
    """
    dtmp = data.copy()
    mode = dtmp.argmax()
    hmax = dtmp[mode]          # 1st max
    dtmp[mode] = 0
    smax = dtmp[dtmp.argmax()] # 2nd max
    dtmp[mode] = hmax
    if (hmax > smax * 2) and (smax != 0):
        dtmp[mode] = int(smax * 1.5)
    return IJIsoData(dtmp)
#------------------------------------------------------------------------------
#
def IJIsoData(data):
    """
    This is the original ImageJ IsoData implementation, here for backward
    compatibility.
    """
    maxValue = len(data) - 1
    count0 = data[0]
    data[0] = 0 #set to zero so erased areas aren't included
    countMax = data[maxValue]
    data[maxValue] = 0
    amin = 0
    while (data[amin] == 0) and (amin < maxValue):
        amin += 1
    amax = maxValue
    while (data[amax] == 0) and (amax > 0):
        amax -= 1
    if amin >= amax:
        data[0] = count0
        data[maxValue] = countMax
        return len(data) / 2

    movingIndex = amin
    while True:
        sum1 = sum2 = sum3 = sum4 = 0.0
        for i in range(amin, movingIndex+1):
            sum1 += float(i) * data[i]
            sum2 += data[i]
        for i in range(movingIndex+1, amax+1):
            sum3 += float(i) * data[i]
            sum4 += data[i]
        result = (sum1 / sum2 + sum3 / sum4) / 2.0
        movingIndex += 1
        if not (((movingIndex+1) <= result) and (movingIndex < amax-1)):
            break
    data[0] = count0
    data[maxValue] = countMax
    return int(np.round(result))
#------------------------------------------------------------------------------
#
def Li(data):
    """
    Implements Li's Minimum Cross Entropy thresholding method
    This implementation is based on the iterative version (Ref. 2) of the
    algorithm.
     1) Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding" 
        Pattern Recognition, 26(4): 617-625
     2) Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum 
        Cross Entropy Thresholding"Pattern Recognition Letters, 18(8): 771-776
     3) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding 
        Techniques and Quantitative Performance Evaluation" Journal of 
        Electronic Imaging, 13(1): 146-165 
        http://citeseer.ist.psu.edu/sezgin04survey.html
    """
    tolerance = 0.5 # threshold tolerance
    num_pixels = float(data.sum())
    # Calculate the mean gray-level
    mean = 0.0; # mean gray-level in the image
    for ih in range(0 + 1, 256): #0 + 1?
        mean += float(ih) * data[ih]
    mean /= num_pixels
    # Initial estimate */
    new_thresh = mean
    while True:
        old_thresh = new_thresh
        threshold = int(old_thresh + 0.5)   # range
        # Calculate the means of background and object pixels */
        # Background
        sum_back = 0.0 # sum of the background pixels at a given threshold
        num_back = 0.0 # number of background pixels at a given threshold
        for ih in range(0, threshold+1):
            sum_back += float(ih) * data[ih]
            num_back += data[ih]
        # mean of the background pixels at a given threshold
        mean_back = 0.0 if num_back == 0 else (sum_back / num_back)
        # Object 
        sum_obj = 0.0 # sum of the object pixels at a given threshold
        num_obj = 0.0 # number of object pixels at a given threshold
        for ih in range(threshold + 1, 256):
            sum_obj += float(ih) * data[ih]
            num_obj += data[ih]
        # mean of the object pixels at a given threshold
        mean_obj = 0.0 if num_obj == 0 else (sum_obj / num_obj)

        # Calculate the new threshold: Equation (7) in Ref. 2
        temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))

        if temp < -2.220446049250313E-16:
            new_thresh = int(temp - 0.5)
        else:
            new_thresh = int(temp + 0.5)
        # Stop the iterations when the difference between the
        # new and old threshold values is less than the tolerance
        if not (np.abs(new_thresh - old_thresh) > tolerance):
            break
    return threshold
#------------------------------------------------------------------------------
#
def MaxEntropy(data):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for
    Gray-Level Picture Thresholding Using the Entropy of the Histogram"
    Graphical Models and Image Processing, 29(3): 273-285
    """
    total = float(data.sum())
    hnorm = data / total # normalized histogram
    P1 = hnorm.cumsum() # cumulative normalized histogram
    P2 = 1.0 - P1
    # Determine the first & last non-zero bin
    first_bin = np.where(P1 >= 2.220446049250313E-16)[0][0]
    last_bin = np.where(P2 >= 2.220446049250313E-16)[0][-1]
    # Calculate the total entropy each gray-level
    # and find the threshold that maximizes it 
    threshold = -1
    max_ent = np.finfo(np.float).min # max entropy
    for it in range(first_bin, last_bin+1):
        # Entropy of the background pixels
        ent_back = 0.0 # entropy of the background pixels at a given threshold
        for ih in range(0, it+1):
            if data[ih] != 0:
                ent_back -= (hnorm[ih] / P1[it]) * np.log(hnorm[ih] / P1[it])
        # Entropy of the object pixels
        ent_obj = 0.0 # entropy of the object pixels at a given threshold
        for ih in range(it+1, 256):
            if data[ih] != 0:
                ent_obj -= (hnorm[ih] / P2[it]) * np.log(hnorm[ih] / P2[it])
        # Total entropy
        tot_ent = ent_back + ent_obj
        if max_ent < tot_ent:
            max_ent = tot_ent
            threshold = it

    return threshold
#------------------------------------------------------------------------------
#
def Mean(data):
    """
    C. A. Glasbey, "An analysis of histogram-based thresholding algorithms,"
    CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.

    The threshold is the mean of the greyscale data
    """
    isum = 0.0  # sum of intensity
    for i in range(1, 256):
        isum += float(i) * data[i]
    return int(isum / data.sum())
#------------------------------------------------------------------------------
#
def MinError(data):
    """
    Kittler and J. Illingworth, "Minimum error thresholding," Pattern Recognition,
    vol. 19, pp. 41-47, 1986.

    C. A. Glasbey, "An analysis of histogram-based thresholding algorithms,
    "CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.

    Ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
    Original Matlab code Copyright (C) 2004 Antti Niemisto
    See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
    and the original Matlab code.
    """
    def A(y, j):
        if j >= len(y):
            j = len(y) - 1
        return float(y[0:j+1].sum())
    def B(y, j):
        if j >= len(y):
            j = len(y) - 1
        x = 0.0;
        for i in np.arange(0,j+1):
            x += float(i) * y[i]
        return x
    def C(y, j):
        if j >= len(y):
            j = len(y) - 1
        x = 0.0
        for i in np.arange(0, j+1):
            x += float(i) * i * y[i]
        return x

    # Initial estimate for the threshold is found with the MEAN algorithm.
    threshold = Mean(data)
    Tprev = -2
    L = len(data) - 1
    AL = A(data, L)
    BL = B(data, L)
    CL = C(data, L)
    while threshold != Tprev:
        # Calculate some statistics.
        At = A(data, threshold)
        Bt = B(data, threshold)
        Ct = C(data, threshold)
        mu = Bt / At
        nu = (BL - Bt) / (AL - At)
        p = At / AL
        q = 1.0 - p  #(AL - At) / AL
        sigma2 = Ct / At - (mu * mu)
        tau2 = (CL - Ct) / (AL - At) - (nu * nu)

        #The terms of the quadratic equation to be solved.
        w0 = 1.0 / sigma2 - 1.0 / tau2
        w1 = mu / sigma2 - nu / tau2
        w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + \
             np.log10((sigma2 * q * q) / (tau2 * p * p))

        #If the next threshold would be imaginary, return with the current one.
        sqterm = (w1 * w1) - w0 * w2
        if sqterm < 0:
            print("MinError(I): not converging.")
            return threshold

        #The updated threshold is the integer part of the solution of the quadratic equation.
        Tprev = threshold
        temp = (w1 + np.sqrt(sqterm)) / w0

        if np.isnan(temp):
            print("MinError(I): NaN, not converging.")
            threshold = Tprev
        else:
            threshold = int(temp)
    return threshold
#------------------------------------------------------------------------------
#
def Minimum(data):
    """
    J. M. S. Prewitt and M. L. Mendelsohn, "The analysis of cell images," in
    Annals of the New York Academy of Sciences, vol. 128, pp. 1035-1053, 1966.
    ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
    Original Matlab code Copyright (C) 2004 Antti Niemisto
    See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
    and the original Matlab code.

    Assumes a bimodal histogram. The histogram needs is smoothed (using a
    running average of size 3, iteratively) until there are only two local maxima.
    Threshold t is such that yt-1 > yt <= yt+1.
    Images with histograms having extremely unequal peaks or a broad and
    flat valleys are unsuitable for this method.
    """
    niter = 0
    threshold = -1
    iHisto = data.astype(np.float)
    tHisto = iHisto.copy()
    while not _isBimodal(iHisto):
        # Smooth with a 3 point running mean filter
        for i in range(1, 255):
            tHisto[i]= (iHisto[i-1] + iHisto[i] + iHisto[i+1]) / 3
        tHisto[0] = (iHisto[0] + iHisto[1]) / 3 #0 outside
        tHisto[255] = (iHisto[254] + iHisto[255]) / 3 #0 outside
        iHisto = tHisto
        niter += 1
        if niter > 10000:
            print("Minimum: threshold not found after 10000 iterations.")
            return -1
    # The threshold is the minimum between the two peaks.
    for i in range(1, 255):
        if (iHisto[i-1] > iHisto[i]) and (iHisto[i+1] >= iHisto[i]):
            threshold = i
            break
    return threshold
#------------------------------------------------------------------------------
#
def Moments(data):
    """
    W. Tsai, "Moment-preserving thresholding: a new approach," Computer Vision,
    Graphics, and Image Processing, vol. 29, pp. 377-393, 1985.
    Ported to ImageJ plugin by G.Landini from the the open source project FOURIER 0.8
    by  M. Emre Celebi , Department of Computer Science,  Louisiana State University in Shreveport
    Shreveport, LA 71115, USA
      http://sourceforge.net/projects/fourier-ipal
      http://www.lsus.edu/faculty/~ecelebi/fourier.htm
    """
    m0 = 1.0
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    threshold = -1
    total = float(data.sum());
    histo = data / total #normalised histogram

    # Calculate the first, second, and third order moments */
    for di in np.arange(0, 256.):
        m1 += di * histo[di]
        m2 += di * di * histo[di]
        m3 += di * di * di * histo[di]

    # First 4 moments of the gray-level image should match the first 4 moments
    # of the target binary image. This leads to 4 equalities whose solutions 
    # are given in the Appendix of Ref. 1 
    cd = m0 * m2 - m1 * m1
    c0 = ( -m2 * m2 + m1 * m3 ) / cd
    c1 = ( m0 * -m3 + m2 * m1 ) / cd
    z0 = 0.5 * ( -c1 - np.sqrt ( c1 * c1 - 4.0 * c0 ) )
    z1 = 0.5 * ( -c1 + np.sqrt ( c1 * c1 - 4.0 * c0 ) )
    # Fraction of the object pixels in the target binary image
    p0 = ( z1 - m1 ) / ( z1 - z0 )

    # The threshold is the gray-level closest  
    # to the p0-tile of the normalized histogram 
    csum = 0.0
    for i in range(0, 256):
        csum += histo[i]
        if csum > p0:
            threshold = i
            break
    return threshold
#------------------------------------------------------------------------------
#
def Otsu(data):
    """
    Otsu's threshold algorithm
    C++ code by Jordan Bevik <Jordan.Bevic@qtiworld.com>
    ported to ImageJ plugin by G.Landini
    """
    # Initialize values:
    L = 256
    N = float(data.sum())   # Total number of data points
    S = 0.0                 # Total histogram intensity
    for k in range(0, L):
        S += k * data[k]
    # The total intensity for all histogram points <=k
    Sk = 0.0
    # The entry for zero intensity
    N1 = float(data[0])
    # The current Between Class Variance and maximum BCV
    BCV = BCVmax = 0.0
    # Optimal threshold
    kStar = 0
    # Look at each possible threshold value,
    # calculate the between-class variance, and decide if it's a max
    for k in range(1, L-1): # No need to check endpoints k = 0 or k = L-1
        Sk += k * data[k]
        N1 += data[k]

        # The float casting here is to avoid compiler warning about loss of
        # precision and will prevent overflow in the case of large saturated
        # images
        denom = N1 * (N - N1) # Maximum value of denom is N^2/4 = approx. 3E10

        if denom != 0:
            # Float here is to avoid loss of precision when dividing
            num = (N1 / N) * S - Sk  # Maximum value of num = 255*N = approx 8E7
            BCV = (num * num) / denom
        else:
            BCV = 0

        if BCV >= BCVmax: # Assign the best threshold found so far
            BCVmax = BCV
            kStar = k
    # kStar += 1;  # Use QTI convention that intensity -> 1 if intensity >= k
    # (the algorithm was developed for I-> 1 if I <= k.)
    return kStar
#------------------------------------------------------------------------------
#
def Percentile(data, ptile = 0.5):
    """
    W. Doyle, "Operation useful for similarity-invariant pattern recognition,"
    Journal of the Association for Computing Machinery, vol. 9,pp. 259-267, 1962.
    ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
    Original Matlab code Copyright (C) 2004 Antti Niemisto
    See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
    and the original Matlab code.
    - ptile: default fraction of foreground pixels
    """
    threshold = -1
    avec = np.zeros(256)
    total = float(data.sum())
    temp = 1.0
    for i in range(0, 256):
        avec[i] = np.abs((data[0:i+1].sum() / total) - ptile)
        if avec[i] < temp:
            temp = avec[i]
            threshold = i
    return threshold
#------------------------------------------------------------------------------
#
def RenyiEntropy(data):
    """
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for
    Gray-Level Picture Thresholding Using the Entropy of the Histogram"
    Graphical Models and Image Processing, 29(3): 273-285
    """
    P1 = np.ndarray(256) # cumulative normalized histogram
    P2 = np.ndarray(256)
    total = float(data.sum())
    hnorm = data / total # normalized histogram

    P1[0] = hnorm[0];
    P2[0] = 1.0 - P1[0];
    for ih in range(1, 256):
        P1[ih]= P1[ih-1] + hnorm[ih];
        P2[ih]= 1.0 - P1[ih];
    # Determine the first non-zero bin
    first_bin = 0;
    for ih in range(0, 256):
        if np.abs(P1[ih]) >= 2.220446049250313E-16:
            first_bin = ih;
            break;
    # Determine the last non-zero bin
    last_bin = 255;
    for ih in range(255, first_bin-1, -1):
        if np.abs(P2[ih]) >= 2.220446049250313E-16:
            last_bin = ih;
            break;
    # Maximum Entropy Thresholding - BEGIN
    # ALPHA = 1.0
    # Calculate the total entropy each gray-level
    # and find the threshold that maximizes it 
    threshold = 0; # was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0; # max entropy
    for it in range(first_bin, last_bin+1):
        # Entropy of the background pixels
        ent_back = 0.0; # entropy of the background pixels at a given threshold
        for ih in range(0, it+1):
            if data[ih] != 0:
                ent_back -= (hnorm[ih] / P1[it]) * np.log(hnorm[ih] / P1[it]);
        # Entropy of the object pixels
        ent_obj = 0.0; # entropy of the object pixels at a given threshold
        for ih in range(it + 1, 256):
            if data[ih] != 0:
                ent_obj -= (hnorm[ih]/P2[it]) * np.log(hnorm[ih]/P2[it]);
        # Total entropy */
        tot_ent = ent_back + ent_obj;
        # IJ.log(""+max_ent+"  "+tot_ent);
        if  max_ent < tot_ent:
            max_ent = tot_ent;
            threshold = it;
    t_star2 = threshold;

    # Maximum Entropy Thresholding - END
    threshold = 0; #was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0;
    alpha = 0.5 # alpha parameter of the method
    term = 1.0 / ( 1.0 - alpha );
    for it in range(first_bin, last_bin+1):
        # Entropy of the background pixels
        ent_back = 0.0;
        for ih in range(0, it+1):
            ent_back += np.sqrt(hnorm[ih] / P1[it]);
        # Entropy of the object pixels
        ent_obj = 0.0;
        for ih in range(it + 1, 256):
            ent_obj += np.sqrt(hnorm[ih] / P2[it]);
        # Total entropy
        tot_ent = term * (np.log(ent_back * ent_obj) if (ent_back * ent_obj) > 0.0 else 0.0);
        if tot_ent > max_ent:
            max_ent = tot_ent;
            threshold = it;

    t_star1 = threshold;

    threshold = 0; #was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0;
    alpha = 2.0;
    term = 1.0 / ( 1.0 - alpha );
    for it in range(first_bin, last_bin+1):
        # Entropy of the background pixels
        ent_back = 0.0;
        for ih in range(0, it+1):
            ent_back += (hnorm[ih] * hnorm[ih]) / (P1[it] * P1[it]);
        # Entropy of the object pixels
        ent_obj = 0.0;
        for ih in range(it + 1, 256):
            ent_obj += (hnorm[ih] * hnorm[ih]) / (P2[it] * P2[it]);
        # Total entropy
        tot_ent = term * (np.log(ent_back * ent_obj) if (ent_back * ent_obj) > 0.0 else 0.0);
        if tot_ent > max_ent:
            max_ent = tot_ent;
            threshold = it;

    t_star3 = threshold;
    # Sort t_star values
    if t_star2 < t_star1:
        tmp_var = t_star1;
        t_star1 = t_star2;
        t_star2 = tmp_var;
    if t_star3 < t_star2:
        tmp_var = t_star2;
        t_star2 = t_star3;
        t_star3 = tmp_var;
    if t_star2 < t_star1:
        tmp_var = t_star1;
        t_star1 = t_star2;
        t_star2 = tmp_var;
    # Adjust beta values
    if np.abs(t_star1 - t_star2) <= 5:
        if np.abs ( t_star2 - t_star3 ) <= 5:
            beta1 = 1;
            beta2 = 2;
            beta3 = 1;
        else:
            beta1 = 0;
            beta2 = 1;
            beta3 = 3;
    else:
        if np.abs(t_star2 - t_star3) <= 5:
            beta1 = 3;
            beta2 = 1;
            beta3 = 0;
        else:
            beta1 = 1;
            beta2 = 2;
            beta3 = 1;
    #IJ.log(""+t_star1+" "+t_star2+" "+t_star3);
    # Determine the optimal threshold value
    omega = P1[t_star3] - P1[t_star1];
    opt_threshold = t_star1 * (P1[t_star1] + 0.25 * omega * beta1) +
                    0.25 * t_star2 * omega * beta2  +
                    t_star3 * (P2[t_star3] + 0.25 * omega * beta3);

    return int(opt_threshold);
#------------------------------------------------------------------------------
#
def Shanbhag(data):
    """
    Shanhbag A.G. (1994) "Utilization of Information Measure as a Means of
    Image Thresholding" Graphical Models and Image Processing, 56(5): 414-419
    """
    P1 = np.ndarray(256) # cumulative normalized histogram
    P2 = np.ndarray(256)

    total = float(data.sum())
    hnorm = data / total; # normalized histogram

    P1[0] = hnorm[0];
    P2[0] = 1.0 - P1[0];
    for ih in range(1, 256):
        P1[ih]= P1[ih-1] + hnorm[ih];
        P2[ih]= 1.0 - P1[ih];
    # Determine the first non-zero bin
    first_bin = 0;
    for ih in range(0, 256):
        if np.abs(P1[ih]) >= 2.220446049250313E-16:
            first_bin = ih;
            break;
    # Determine the last non-zero bin
    last_bin = 255;
    for ih in range(255, first_bin-1, -1):
        if np.abs(P2[ih]) >= 2.220446049250313E-16:
            last_bin = ih;
            break;
    # Calculate the total entropy each gray-level
    # and find the threshold that maximizes it 
    threshold = -1;
    min_ent = np.finfo(np.float).max
    for it in range(first_bin, last_bin+1):
        # Entropy of the background pixels
        ent_back = 0.0; # entropy of the background pixels at a given threshold
        term = 0.5 / P1[it];
        for ih in range(1, it+1):
            ent_back -= hnorm[ih] * np.log(1.0 - term * P1[ih - 1]);
        ent_back *= term;
        # Entropy of the object pixels
        ent_obj = 0.0;  # entropy of the object pixels at a given threshold
        term = 0.5 / P2[it];
        for ih in range(it + 1, 256):
            ent_obj -= hnorm[ih] * np.log(1.0 - term * P2[ih]);
        ent_obj *= term;
        # Total entropy
        tot_ent = np.abs(ent_back - ent_obj);
        if tot_ent < min_ent:
            min_ent = tot_ent;
            threshold = it;
    return threshold;
#------------------------------------------------------------------------------
#
def Triangle(data):
    """
    Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
    Automatic Measurement of Sister Chromatid Exchange Frequency,
    Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
    """
    # find min and max
    min2 = 0
    nz = np.where(data > 0)[0]
    amin = nz[0]
    amax = nz[-1]
    if amin > 0: amin -= 1; # line to the (p==0) point, not to data[min]

    # The Triangle algorithm cannot tell whether the data is skewed to one side or another.
    # This causes a problem as there are 2 possible thresholds between the max and the 2 extremes
    # of the histogram.
    # Here I propose to find out to which side of the max point the data is furthest, and use that as
    #  the other extreme.
    for i in range(255, 0, -1):
        if data[i] > 0:
            min2 = i
            break
    if min2 < 255: min2 += 1 # line to the (p==0) point, not to data[min]

    amax = data.argmax()
    #dmax = data[amax]
    # find which is the furthest side
    inverted = False
    if (amax - amin) < (min2 - amax):
        # reverse the histogram
        inverted = True
        left  = 0       # index of leftmost element
        right = 255     # index of rightmost element
        while left < right:
            # exchange the left and right elements
            temp = data[left]
            data[left]  = data[right]
            data[right] = temp
            # move the bounds toward the center
            left += 1
            right -= 1
        amin = 255 - min2
        amax = 255 - amax

    if amin == amax:
        #IJ.log("Triangle:  min == max.");
        return amin

    # describe line by nx * x + ny * y - d = 0
    # nx is just the max frequency as the other point has freq=0
    nx = float(data[amax])   #-min; # data[min]; #  lowest value bmin = (p=0)% in the image
    ny = float(amin - amax)
    d = np.sqrt(nx * nx + ny * ny)
    nx /= d
    ny /= d
    d = nx * amin + ny * data[amin]

    # find split point
    split = amin
    splitDistance = 0.0
    for i in range(amin + 1, amax+1):
        newDistance = nx * i + ny * data[i] - d
        if newDistance > splitDistance:
            split = i
            splitDistance = newDistance
    split -= 1

    if inverted:
        # The histogram might be used for something else, so let's reverse it back
        left  = 0
        right = 255
        while left < right:
            temp = data[left]
            data[left]  = data[right]
            data[right] = temp
            left += 1
            right -= 1
        return 255-split
    return split
#------------------------------------------------------------------------------
#
def Yen(data):
    """
    Implements Yen  thresholding method
     1) Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion 
        for Automatic Multilevel Thresholding" IEEE Trans. on Image 
        Processing, 4(3): 370-378
     2) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding 
        Techniques and Quantitative Performance Evaluation" Journal of 
        Electronic Imaging, 13(1): 146-165
        http://citeseer.ist.psu.edu/sezgin04survey.html
    """
    total = float(data.sum())
    hnorm = data / total # normalized histogram
    P1 = hnorm.cumsum()
    P1_sq = (hnorm ** 2).cumsum()
    P2_sq = (hnorm[::-1] ** 2).cumsum()[::-1]
    P2_sq -= P2_sq[0]
#    P2_sq = np.ndarray(256)
#    P2_sq[255] = 0.0;
#    for ih in range(254, -1, -1):
#        P2_sq[ih] = P2_sq[ih + 1] + hnorm[ih + 1] * hnorm[ih + 1];

    # Find the threshold that maximizes the criterion
    threshold = -1
    max_crit = np.finfo(np.float).min
    for it in range(0, 256):
        s1s2 = P1_sq[it] * P2_sq[it]
        p1p1 = P1[it] * (1.0 - P1[it])
        crit = -1.0 * (np.log(s1s2) if s1s2 > 0.0 else 0.0) + \
                2.0 * (np.log(p1p1) if p1p1 > 0.0 else 0.0)
        if crit > max_crit:
            max_crit = crit;
            threshold = it;
    return threshold;
#------------------------------------------------------------------------------
# List of all binarization functions
Fx = [              \
    DefaultIsoData, \
    Huang,          \
    IJIsoData,      \
    Intermodes,     \
    IsoData,        \
    Li,             \
    MaxEntropy,     \
    Mean,           \
    MinError,       \
    Minimum,        \
    Moments,        \
    Otsu,           \
    Percentile,     \
    RenyiEntropy,   \
    Shanbhag,       \
    Triangle,       \
    Yen             \
]
#------------------------------------------------------------------------------
# Excute binarization by method id#
#------------------------------------------------------------------------------
def Execute(method, histogram):
    try:
        return Fx[method](histogram)
    except:
        pass
    return 0
#------------------------------------------------------------------------------
# Test code
#------------------------------------------------------------------------------
if __name__ == "__main__":
    from MIVFile import MIV
    d = MIV(r'D:/Data/Distortion.Correction/Image.Stacks/stack_2_1.miv')
    d = d.data[201,:,:].ravel().copy()
    h = np.histogram(d, bins=np.arange(257))[0]
    for name in dir(Methods):
        if name[0:2] == '__':
            continue
        m = eval('Methods.'+name)
        print name, ' = ', Execute(m, h)
