import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.color import rgb2gray
from skimage.util import img_as_float32
from guassiandeconv import gaussian_kernel, forward_model, tikhonov_deconv

k_size = 21
k_sigma = 2.5
sigma_i = 0.02
lam = 0.02
x_true = img_as_float32(data.camera())  # shape (512,512), range [0,1]
psf = gaussian_kernel(size=k_size, sigma=k_sigma)
y, _ = forward_model(x_true, psf, sigma_noise=sigma_i, seed=42)
x_tik = tikhonov_deconv(y, psf, lam=lam)

# show Tikhonov result
plt.figure(figsize=(5, 5))
plt.imshow(x_tik, cmap="gray", vmin=0, vmax=1)
plt.title(f"Tikhonov deconvolution (lambda={lam},sigma={sigma_i})")
plt.axis("off")
plt.tight_layout()
plt.show()