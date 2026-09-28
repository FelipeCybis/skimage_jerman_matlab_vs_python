# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-image @ git+https://github.com/FelipeCybis/scikit-image@feat/add-jerman-filter",
# ]
# ///
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import jerman

retina = rgb2gray(data.retina())

# Same params as the MATLAB run: sigmas 0.5:0.5:2.5, tau=1, brightondark=false
V = jerman(retina, sigmas=[0.5, 1.0, 1.5, 2.0, 2.5], tau=1.0, black_ridges=True)
np.save("retina_jerman_skimage_implementation.npy", V)
