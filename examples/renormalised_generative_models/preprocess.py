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
    upsample images from (d x d) to (l x l) dimensions using bilinear interpolation
    """
    def resize_image(img):
        return jax.image.resize(img, (l, l), method='bilinear')
    
    upsampled = vmap(resize_image)(images)

    return upsampled


def gaussian_smooth(images: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
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
    Apply histogram equalization to enhance image contrast.
    """
    def equalize_single(img):
        flat = img.flatten()
        hist, _ = jnp.histogram(flat, bins=256, range=(0, 255))
        cdf = jnp.cumsum(hist)
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
        indices = jnp.clip(flat.astype(jnp.int32), 0, 255)
        return cdf_normalized[indices].reshape(img.shape)

    return vmap(equalize_single)(images)

def convert_to_rgb(images: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([images, images, images], axis=1)


@jax.jit(static_argnames=['target_size', 'sigma'])
def preprocess(images: jnp.ndarray, target_size: int = 32, sigma: float = 0.3) -> jnp.ndarray:
    """
    Apply full preprocessing pipeline to images.

    Args:
        images: Array of shape (N, H, W) containing N grayscale images
        target_size: Output spatial dimension after upsampling
        sigma: Standard deviation for Gaussian smoothing

    Returns:
        Preprocessed images with shape (N, 3, target_size, target_size)
    """
    images = gaussian_smooth(images, sigma=sigma)
    images = histogram_equalisation(images)
    images = upsample(images, l=target_size)
    images = convert_to_rgb(images)
    return images