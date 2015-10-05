# vipy - A 3D image processing framework in Python
------------------------------------------------------

This framework allows to work with volume/stack images easily. Most of
common filters are supported in this framework, such as mean, median,
sigma, laplace, anisotropic, gaussian, local normalization, homomorphic,
kalman (plan base)... 3D morphology operations for both binary and gray
image volumes are also included.

	$ python
	>>> from MIVFile import MIV
	>>> 
	>>> vol = MIV("stack_1_1.miv")
	>>> vmf = vol.Filter.Mean([3]*3)	# Apply mean filter 3x3x3
	>>> vmf.Save("stack_1_1.af3.miv")	# Save to new file
	>>>
	>>> # Quick load, apply median filter 5x5x5 and save result
	>>> MIV("stack_1_1.miv").Filter.Median(5).Save("stack_1_1.mf5.miv")
	>>>

Load a TIFF file, resize volume and save to MIV format:

	>>> from TIVFile import TIV
	>>> vol = TIV("stack_2_1.tif")
	>>> big = vol.Rescale([1.75]*3, mode='cubic')
	>>> small = vol.Rescale(rsize=[128]*3, mode='lanczos')
	>>> big.Promote(MIV)	# Switch to MIV format
	>>> big.Save("stack_2_1.big175.miv")

The framework also provides a quick and simple algorithm to apply
segmentation (binarization) to a volume:

	>>> from QSeg import *
	>>> MIV_Seg('D:/Data/vessel.sample.miv')
	>>> SegBulk('D:/Data', 'D:/Data/Result', filter='ln')

Besides, it provides a simple way to use C/C++ functions compiled as
shared library (.dll, .so, .dylib) via ExtLib class, VolFilters class
is an example of this mechanism.

