# deblur_baseline_v0.py
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.color import rgb2gray
from skimage.util import img_as_float32


# -----------------------------
# 1) Utils: kernel + FFT helpers
# -----------------------------
def gaussian_kernel(size=21, sigma=2.5):
    """Create a 2D Gaussian PSF kernel, normalized to sum=1."""
    assert size % 2 == 1, "Kernel size should be odd."
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k) #energy =1, the energy of PSF should equal to the energy of delta function
    return k.astype(np.float32)


def psf2otf(psf, shape):
    """
    Convert a spatial PSF to OTF (FFT of padded+shifted PSF), so that:
    circular_conv(x, psf) == ifft2( fft2(x) * OTF )
    """
    #in order to apply frequency filtering, the size of PSF spectrum and the size of image spectrm should be the same
    psf_pad = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf

    # Shift PSF center to (0,0) for correct circular convolution in FFT domain
    psf_pad = np.roll(psf_pad, -kh // 2, axis=0)
    psf_pad = np.roll(psf_pad, -kw // 2, axis=1)

    return np.fft.fft2(psf_pad)


def psf2otf_raw_center(psf, shape):
    """
    "Raw" variant for comparison only:
    put PSF directly at image center, then FFT (without ifftshift-style correction).
    """
    psf_pad = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape

    h, w = shape
    r0 = h // 2 - kh // 2
    c0 = w // 2 - kw // 2
    psf_pad[r0:r0 + kh, c0:c0 + kw] = psf

    return np.fft.fft2(psf_pad)


def circular_conv2d(x, psf):
    """2D circular convolution using FFT."""
    # typical circular convolution,when you use two same size frequency domains times each other
    # this is called cicular convolution.
    K = psf2otf(psf, x.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(x) * K)).astype(np.float32)


def circular_conv2d_raw_center(x, psf):
    """Circular convolution using the raw-center PSF FFT for visual comparison."""
    K_raw = psf2otf_raw_center(psf, x.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(x) * K_raw)).astype(np.float32)


# -----------------------------
# 2) Forward model: blur + noise
# -----------------------------
def forward_model(x, psf, sigma_noise, seed=0):
    """
    y = k*x + n, n ~ N(0, sigma^2)
    x is assumed in [0,1].
    """
    rng = np.random.default_rng(seed)
    y_blur = circular_conv2d(x, psf)
    n = rng.normal(0.0, sigma_noise, size=x.shape).astype(np.float32)
    y = y_blur + n
    return y, y_blur


# -----------------------------
# 3) Reconstruction: Tikhonov (closed-form)
# -----------------------------
def tikhonov_deconv(y, psf, lam):
    """
    argmin_x 0.5||k*x - y||^2 + 0.5*lam||x||^2
    Closed-form in Fourier domain:
    X = conj(K)*Y / (|K|^2 + lam)
    """
    K = psf2otf(psf, y.shape)
    Y = np.fft.fft2(y)
    X = np.conj(K) * Y / (np.abs(K) ** 2 + lam)
    x_hat = np.real(np.fft.ifft2(X)).astype(np.float32)
    return np.clip(x_hat, 0.0, 1.0)


# -----------------------------
# 4) Reconstruction: Gradient Descent (+ optional early stopping)
# -----------------------------
def gd_deconv(y, psf, lam, x_true=None, max_iters=200, patience=20):
    """
    Gradient descent on:
      f(x)=0.5||k*x - y||^2 + 0.5*lam||x||^2

    With circular convolution:
      grad = k^T*(k*x - y) + lam*x
    where k^T corresponds to conj(K) in Fourier domain.

    Step size alpha chosen via Lipschitz bound:
      L = max(|K|^2) + lam, alpha = 1/L

    Early stopping (optional):
      if x_true provided, stop when PSNR doesn't improve for 'patience' iters.
    """
    K = psf2otf(psf, y.shape)
    Y = np.fft.fft2(y)

    L = float(np.max(np.abs(K) ** 2).real + lam)
    alpha = 1.0 / L

    x = y.copy().astype(np.float32)  # init
    best_psnr = -np.inf
    best_x = x.copy()
    bad_count = 0

    psnr_trace = []

    for it in range(max_iters):
        X = np.fft.fft2(x)
        resid_f = K * X - Y                        # FFT(k*x - y)
        grad_data = np.real(np.fft.ifft2(np.conj(K) * resid_f))  # k^T*(k*x - y)
        grad = grad_data + lam * x

        x = x - alpha * grad
        x = np.clip(x, 0.0, 1.0)

        if x_true is not None:
            p = peak_signal_noise_ratio(x_true, x, data_range=1.0)
            psnr_trace.append(p)
            if p > best_psnr + 1e-6:
                best_psnr = p
                best_x = x.copy()
                bad_count = 0
            else:
                bad_count += 1
            if bad_count >= patience:
                break

    return best_x if x_true is not None else x, alpha, np.array(psnr_trace, dtype=np.float32)


# -----------------------------
# 5) Visualization
# -----------------------------
def make_figure(x_true, y, x_tik, x_gd, title):
    y_clip = np.clip(y, 0.0, 1.0)
    err_tik = np.abs(x_tik - x_true)
    err_gd = np.abs(x_gd - x_true)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(x_true, cmap="gray")
    axes[0].set_title("GT x")
    axes[0].axis("off")

    axes[1].imshow(y_clip, cmap="gray")
    axes[1].set_title("Input y (blur+noise)")
    axes[1].axis("off")

    axes[2].imshow(x_tik, cmap="gray")
    axes[2].set_title("Tikhonov x̂")
    axes[2].axis("off")

    axes[3].imshow(x_gd, cmap="gray")
    axes[3].set_title("GD x̂")
    axes[3].axis("off")

    axes[4].imshow(err_tik, cmap="gray")
    axes[4].set_title("|x̂_tik - x|")
    axes[4].axis("off")

    axes[5].imshow(err_gd, cmap="gray")
    axes[5].set_title("|x̂_gd - x|")
    axes[5].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_raw_vs_nonraw_figure(x_true, y_nonraw, y_raw, title):
    y_nonraw_clip = np.clip(y_nonraw, 0.0, 1.0)
    y_raw_clip = np.clip(y_raw, 0.0, 1.0)
    diff = np.abs(y_nonraw_clip - y_raw_clip)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(x_true, cmap="gray")
    axes[0].set_title("GT x")
    axes[0].axis("off")

    axes[1].imshow(y_nonraw_clip, cmap="gray")
    axes[1].set_title("Non-raw conv")
    axes[1].axis("off")

    axes[2].imshow(y_raw_clip, cmap="gray")
    axes[2].set_title("Raw-center conv")
    axes[2].axis("off")

    axes[3].imshow(diff, cmap="magma")
    axes[3].set_title("|non-raw - raw|")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# -----------------------------
# 6) Solver Pipeline
# -----------------------------
def run_deblur_pipeline(sigma=0.01, k_size=21, k_sigma=2.5, lam=None, outdir="outputs", compare_raw=True, log_fn=None):
    """
    Run deblurring experiments and save figures.
    This is the core solving pipeline used by both script and UI.
    """
    if log_fn is None:
        log_fn = print

    os.makedirs(outdir, exist_ok=True)

    # Use a built-in test image to keep it self-contained
    x_true = img_as_float32(data.camera())  # shape (512,512), range [0,1]
    psf = gaussian_kernel(size=k_size, sigma=k_sigma)

    if compare_raw:
        y_nonraw = circular_conv2d(x_true, psf)
        y_raw = circular_conv2d_raw_center(x_true, psf)
        psnr_nonraw = peak_signal_noise_ratio(x_true, np.clip(y_nonraw, 0, 1), data_range=1.0)
        psnr_raw = peak_signal_noise_ratio(x_true, np.clip(y_raw, 0, 1), data_range=1.0)
        mse_between = mean_squared_error(np.clip(y_nonraw, 0, 1), np.clip(y_raw, 0, 1))

        title_cmp = (
            f"Raw vs Non-raw conv | PSNR(non-raw)={psnr_nonraw:.2f}, "
            f"PSNR(raw)={psnr_raw:.2f}, MSE(between)={mse_between:.4e}"
        )
        fig_cmp = make_raw_vs_nonraw_figure(x_true, y_nonraw, y_raw, title_cmp)
        cmp_path = os.path.join(outdir, "compare_raw_vs_nonraw.png")
        fig_cmp.savefig(cmp_path, dpi=150)
        plt.close(fig_cmp)
        log_fn(f"Saved: {os.path.abspath(cmp_path)}")

    if lam is None:
        experiments = [
            {"sigma": sigma, "lam": 1e-4},
            {"sigma": sigma, "lam": 1e-2},
        ]
    else:
        experiments = [{"sigma": sigma, "lam": float(lam)}]

    results = []
    log_fn("")
    log_fn("Results:")
    log_fn("idx  sigma    lambda     PSNR_in  PSNR_Tik  PSNR_GD   MSE_Tik     MSE_GD     alpha      GD_iters")

    for i, cfg in enumerate(experiments):
        sigma_i = cfg["sigma"]
        lam = cfg["lam"]

        y, _ = forward_model(x_true, psf, sigma_noise=sigma_i, seed=42 + i)
        x_tik = tikhonov_deconv(y, psf, lam=lam)
        x_gd, alpha, psnr_trace = gd_deconv(y, psf, lam=lam, x_true=x_true, max_iters=300, patience=30)

        psnr_in = peak_signal_noise_ratio(x_true, np.clip(y, 0, 1), data_range=1.0)
        psnr_t = peak_signal_noise_ratio(x_true, x_tik, data_range=1.0)
        psnr_g = peak_signal_noise_ratio(x_true, x_gd, data_range=1.0)
        mse_t = mean_squared_error(x_true, x_tik)
        mse_g = mean_squared_error(x_true, x_gd)

        title = (
            f"Exp{i}: sigma={sigma_i:.4f}, lambda={lam:.1e} | "
            f"PSNR(in)={psnr_in:.2f}, T={psnr_t:.2f}, GD={psnr_g:.2f}"
        )
        fig = make_figure(x_true, y, x_tik, x_gd, title)
        fname = os.path.join(outdir, f"exp{i}_sigma{sigma_i:.4f}_lam{lam:.1e}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)

        row = (
            f"{i:<4d}{sigma_i:<8.4f}{lam:<10.1e}{psnr_in:<9.2f}{psnr_t:<10.2f}{psnr_g:<9.2f}"
            f"{mse_t:<12.4e}{mse_g:<12.4e}{alpha:<11.3e}{len(psnr_trace):<8d}"
        )
        log_fn(row)
        log_fn(f"Saved: {os.path.abspath(fname)}")

        results.append((i, sigma_i, lam, psnr_in, psnr_t, psnr_g, mse_t, mse_g, alpha, len(psnr_trace)))

    log_fn("")
    log_fn(f"Done. Outputs in: {os.path.abspath(outdir)}")
    return results


# -----------------------------
# 7) Main: UI entry only
# -----------------------------
def launch_ui():
    """Launch the standalone Tkinter UI module."""
    from guassiandeconv_ui import main as ui_main
    ui_main()


def main():
    launch_ui()


if __name__ == "__main__":
    main()
