"""
SVD-based discretisation of images for Renormalised Generative Models.

Implements the blocking transformation from:
    Friston et al. (2025) "From pixels to planning: scale-free active inference"

Pipeline: partition into patches -> SVD basis learning -> quantisation into discrete bins.
"""

from typing import NamedTuple

from jax import numpy as jnp, vmap


class DiscretiseConfig(NamedTuple):
    group_size: int = 4
    n_components: int = 16
    n_levels: int = 7
    image_size: int = 32
    n_channels: int = 3


class SVDBasis(NamedTuple):
    V: jnp.ndarray          # (n_groups, n_groups, n_features, n_components)
    S: jnp.ndarray          # (n_groups, n_groups, n_components)
    mean: jnp.ndarray       # (n_groups, n_groups, n_features)
    bin_edges: jnp.ndarray  # (n_groups, n_groups, n_components, n_levels+1)


def get_gaussian_weights(config: DiscretiseConfig) -> jnp.ndarray:
    """Radial Gaussian weighting mask for a flattened patch (sigma = group_size)."""
    g = config.group_size
    x = jnp.arange(g) - (g - 1) / 2
    xx, yy = jnp.meshgrid(x, x)
    sigma = float(g)
    w2d = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return jnp.tile(w2d, (config.n_channels, 1, 1)).flatten()


def partition_into_patches(images: jnp.ndarray, config: DiscretiseConfig) -> jnp.ndarray:
    """Partition (N, C, H, W) images into (N, n_groups, n_groups, C, g, g) patches."""
    N, C, _H, _W = images.shape
    g = config.group_size
    n = config.image_size // g
    return images.reshape(N, C, n, g, n, g).transpose(0, 2, 4, 1, 3, 5)


def compute_svd_basis(images: jnp.ndarray, config: DiscretiseConfig) -> SVDBasis:
    """
    Learn SVD basis and bin edges from structure images.

    Args:
        images: (N, C, H, W) preprocessed structure images
        config: discretisation parameters

    Returns:
        SVDBasis with learned projection and quantisation parameters
    """
    g = config.group_size
    n = config.image_size // g
    k = config.n_components
    f = config.n_channels * g * g

    patches = partition_into_patches(images, config)
    N = patches.shape[0]
    flat = patches.reshape(N, n, n, f)

    weights = get_gaussian_weights(config)
    weighted = flat * weights[None, None, None, :]

    mean = weighted.mean(axis=0)
    centred = weighted - mean[None, :, :, :]

    # (N, n, n, f) -> (n*n, N, f) for vmapped SVD
    centred_2d = centred.transpose(1, 2, 0, 3).reshape(n * n, N, f)

    def svd_one(data):
        _U, S, Vh = jnp.linalg.svd(data, full_matrices=False)
        return Vh.T[:, :k], S[:k]

    V_flat, S_flat = vmap(svd_one)(centred_2d)
    V = V_flat.reshape(n, n, f, k)
    S = S_flat.reshape(n, n, k)

    # Project structure images to compute bin edges
    variates = jnp.einsum('bijf,ijfc->bijc', centred, V)  # (N, n, n, k)

    def symmetric_bins(v):
        max_abs = jnp.maximum(jnp.abs(v).max(), 1e-8)
        return jnp.linspace(-max_abs, max_abs, config.n_levels + 1)

    # vmap over (n, n, k) -> bin_edges (n, n, k, n_levels+1)
    bin_edges = vmap(vmap(vmap(symmetric_bins)))(variates.transpose(1, 2, 3, 0))

    return SVDBasis(V=V, S=S, mean=mean, bin_edges=bin_edges)


def encode_images(images: jnp.ndarray, basis: SVDBasis, config: DiscretiseConfig) -> jnp.ndarray:
    """
    Encode images as discrete observations.

    Returns:
        (N, n_groups, n_groups, n_components) integer bin indices in [0, n_levels-1]
    """
    g = config.group_size
    n = config.image_size // g
    f = config.n_channels * g * g

    patches = partition_into_patches(images, config)
    N = patches.shape[0]
    flat = patches.reshape(N, n, n, f)

    weights = get_gaussian_weights(config)
    centred = flat * weights[None, None, None, :] - basis.mean[None, :, :, :]

    variates = jnp.einsum('bijf,ijfc->bijc', centred, basis.V)

    # Discretise
    v_flat = variates.reshape(N, -1)                                # (N, n*n*k)
    e_flat = basis.bin_edges.reshape(-1, basis.bin_edges.shape[-1])  # (n*n*k, n_levels+1)

    def digitize(v, edges):
        return jnp.clip(jnp.searchsorted(edges, v, side='right') - 1, 0, config.n_levels - 1)

    indices = vmap(digitize, in_axes=(1, 0), out_axes=1)(v_flat, e_flat)
    return indices.reshape(N, n, n, config.n_components)


def decode_observations(
    observations: jnp.ndarray, basis: SVDBasis, config: DiscretiseConfig
) -> jnp.ndarray:
    """
    Reconstruct images from discrete observations.

    Args:
        observations: (N, n_groups, n_groups, n_components) integer bin indices

    Returns:
        (N, C, H, W) reconstructed images
    """
    g = config.group_size
    n = config.image_size // g
    C = config.n_channels
    f = C * g * g

    # Bin indices -> bin centres
    centres = (basis.bin_edges[..., :-1] + basis.bin_edges[..., 1:]) / 2  # (n, n, k, n_levels)

    # Gather the centre for each observation
    def gather_centres(obs_ijn, centres_ijn):
        # obs_ijn: (k,) int indices, centres_ijn: (k, n_levels)
        return centres_ijn[jnp.arange(centres_ijn.shape[0]), obs_ijn]

    def gather_image(obs_i):
        # obs_i: (n, n, k)
        return vmap(vmap(gather_centres))(obs_i, centres)

    variates = vmap(gather_image)(observations)  # (N, n, n, k)

    # Unproject
    recon = jnp.einsum('bijc,ijfc->bijf', variates, basis.V)  # (N, n, n, f)
    recon = recon + basis.mean[None, :, :, :]

    # Remove Gaussian weighting
    weights = get_gaussian_weights(config)
    recon = recon / weights[None, None, None, :]

    # Reassemble: (N, n, n, C*g*g) -> (N, n, n, C, g, g) -> (N, C, H, W)
    patches = recon.reshape(-1, n, n, C, g, g)
    return patches.transpose(0, 3, 1, 4, 2, 5).reshape(-1, C, n * g, n * g)
