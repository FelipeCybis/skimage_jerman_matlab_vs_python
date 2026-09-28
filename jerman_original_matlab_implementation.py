# /// script
# requires-python = ">=3.9,<3.11"
# dependencies = [
#     "matlabengine==9.13.*",
#     "numpy",
#     "scikit-image",
# ]
# ///
import os

import matlab.engine
import numpy as np
from skimage import data
from skimage.color import rgb2gray

URL = "https://github.com/timjerman/JermanEnhancementFilter/archive/refs/heads/master.zip"

eng = matlab.engine.start_matlab()
tmp = eng.tempdir()
pkg = os.path.join(tmp, "JermanEnhancementFilter-master")
if not os.path.isdir(pkg):
    eng.unzip(URL, tmp, nargout=0)  # MATLAB unzip takes a URL directly
eng.addpath(pkg, nargout=0)

retina = rgb2gray(data.retina())  # 2D float, MATLAB wants a matrix not RGB

V = eng.vesselness2D(
    matlab.double(retina.tolist()),  # numpy arrays are not auto-converted
    matlab.double([0.5, 1.0, 1.5, 2.0, 2.5]),  # sigmas
    matlab.double([1.0, 1.0]),  # image dimension scaling, not implementing this on skimage
    1.0,  # tau
    False,  # white ridges (opposite to skimage black_ridges argument)
    nargout=1,
)
V = np.asarray(V)
np.save("retina_jerman_original_implementation.npy", V)
