# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-image @ git+https://github.com/FelipeCybis/scikit-image@feat/add-jerman-filter",
# ]
# ///
"""scikit-image ``jerman`` with the MATLAB-style Hessian estimator.

The only substantive difference between the two implementations is how the
Hessian is estimated. MATLAB smooths with a Gaussian (replicate borders) and
then applies central finite differences twice. scikit-image uses Gaussian
derivative kernels. This script patches ``hessian_matrix`` so ``jerman`` uses
scikit-image's own finite-difference path (``use_gaussian_derivatives=False``,
which is Gaussian smoothing followed by ``np.gradient`` twice) with
``mode='nearest'``, the equivalent of MATLAB's replicate border.

Everything else in ``jerman`` runs unchanged.
"""

from functools import wraps
from unittest.mock import patch

import numpy as np
import skimage.feature.corner as corner
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import jerman

_hessian_matrix = corner.hessian_matrix


@wraps(_hessian_matrix)
def hessian_matrix_finite_differences(image, sigma=1, **kwargs):
    kwargs.update(use_gaussian_derivatives=False, mode="nearest")
    return _hessian_matrix(image, sigma, **kwargs)


retina = rgb2gray(data.retina())

# `jerman` does `from ..feature.corner import hessian_matrix` at call time, so
# patching the module attribute is enough.
with patch.object(corner, "hessian_matrix", hessian_matrix_finite_differences):
    V = jerman(retina, sigmas=[0.5, 1.0, 1.5, 2.0, 2.5], tau=1.0, black_ridges=True)

np.save("retina_jerman_skimage_finite_differences.npy", V)
