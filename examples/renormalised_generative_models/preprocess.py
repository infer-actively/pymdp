import numpy as np
import jax
from jax import numpy as jnp
from jax import lax, vmap

from sklearn.datasets import fetch_openml
from typing import Tuple

def load_mnist(cache_dir: str= './mnist_cache') -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=cache_dir, parser='auto')

    x, y = mnist.data, mnist.target

    x = x.values if hasattr(x, 'values') else x
    y = y.values if hasattr(y, 'values') else y

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    x = x.reshape(-1, 28, 28)

    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_test), jnp.array(y_test)


def upsample(images: jnp.ndarray, l=32) -> jnp.ndarray:
    """
    upsample images from (N, d, d) to (N, l, l) using bilinear interpolation
    """
    N = images.shape[0]
    return jax.image.resize(images, (N, l, l), method='bilinear')


def gaussian_smooth(images: jnp.ndarray, sigma: float = 0.85) -> jnp.ndarray:
    """
    smooth a set of images with a Gaussian kernel

    Args:
        images: Array of shape (N, H, W) containing N images
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Smoothed images with the same shape as input
    """
    # Create 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel via outer product
    kernel_2d = jnp.outer(kernel_1d, kernel_1d)

    # Reshape for conv_general_dilated: (out_channels, in_channels, H, W)
    kernel = kernel_2d[None, None, :, :]

    # Add channel dimension to images: (N, H, W) -> (N, 1, H, W)
    images_4d = images[:, None, :, :]

    # Pad images to maintain spatial dimensions
    pad = kernel_size // 2

    # Apply 2D convolution
    smoothed = lax.conv_general_dilated(
        images_4d,
        kernel,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )

    # Remove channel dimension: (N, 1, H, W) -> (N, H, W)
    return smoothed[:, 0, :, :]


@jax.jit
def histogram_equalisation(images: jnp.ndarray) -> jnp.ndarray:
    """
    SPM-style histogram equalization via rank-based Gaussian mapping.

    For each image:
    1. Sort pixel values and get rank indices
    2. Map ranks to exp(-((rank - n)^2) / (S * n)) where S=24, n=num_pixels
    3. Clip to [0, 1] and scale to [0, 255]

    This matches SPM's spm_hist (used in DEM_MNIST_RGM.m).
    """
    S = 24.0

    def equalize_single(img):
        flat = img.flatten()
        n = flat.shape[0]

        # Get rank indices: argsort of argsort gives the rank of each element
        order = jnp.argsort(flat)
        ranks = jnp.zeros_like(order)
        ranks = ranks.at[order].set(jnp.arange(n))

        # SPM's Gaussian mapping: exp(-((rank - n)^2) / (S * n))
        mapped = jnp.exp(-((ranks.astype(jnp.float32) - n) ** 2) / (S * n))

        # Clip to [0, 1] and scale to [0, 255]
        mapped = jnp.clip(mapped, 0.0, 1.0) * 255.0

        return mapped.reshape(img.shape)

    return vmap(equalize_single)(images)


def convert_to_rgb(images: jnp.ndarray) -> jnp.ndarray:
    """Convert (N, H, W) grayscale images to (N, 3, H, W) by repeating channels."""
    return jnp.stack([images, images, images], axis=1)


@jax.jit(static_argnames=['target_size', 'fwhm'])
def preprocess(images: jnp.ndarray, target_size: int = 32, fwhm: float = 2.0) -> jnp.ndarray:
    """
    Apply full preprocessing pipeline to images, matching SPM DEM_MNIST_RGM order.

    Pipeline: upsample(28→32) → gaussian_smooth(sigma) → histogram_equalisation

    Note: even though the main text in the paper says to convert to RGB (3 channels),
    this is not implemented in the SPM demo.

    Args:
        images: Array of shape (N, H, W) containing N grayscale images
        target_size: Output spatial dimension after upsampling
        fwhm: Full width at half maximum for Gaussian smoothing.
              SPM's spm_conv interprets its parameter as FWHM, converting to
              sigma = fwhm / sqrt(8*log(2)) ≈ fwhm / 2.355. Default 2.0
              matches SPM's s=2, giving sigma ≈ 0.85.

    Returns:
        Preprocessed images with shape (N, H, W) grayscale
    """
    sigma = fwhm / np.sqrt(8 * np.log(2.0))
    images = upsample(images, l=target_size)
    images = gaussian_smooth(images, sigma=sigma)
    images = histogram_equalisation(images)
    return images
