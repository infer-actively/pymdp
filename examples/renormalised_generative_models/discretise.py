"""
SVD-based discretisation of images for Renormalised Generative Models.

Implements the blocking transformation from:
    Friston et al. (2025) "From pixels to planning: scale-free active inference"

Pipeline: partition into patches -> SVD basis learning -> quantisation into discrete bins.

Aligned with SPM's DEM_MNIST_RGM.m and spm_rgb2O.m:
- Adaptive SVD mode selection via singular value threshold (spm_svd)
- Nearest-bin assignment using bin centres (not edges)
"""

from typing import NamedTuple

from jax import numpy as jnp, vmap


class DiscretiseConfig(NamedTuple):
    group_size: int = 4        # nd in SPM (tile diameter)
    max_components: int = 16   # mm in SPM (max singular modes)
    sv_threshold: float = 8.0  # su in SPM (1/su = normalized SV cutoff)
    n_levels: int = 7          # nb in SPM (number of discrete bins, forced odd)
    image_size: int = 32
    n_channels: int = 1        # grayscale by default


class SVDBasis(NamedTuple):
    V: jnp.ndarray            # (n_groups, n_groups, n_features, max_components)
    S: jnp.ndarray            # (n_groups, n_groups, max_components)
    mean: jnp.ndarray         # (n_groups, n_groups, n_features)
    bin_centres: jnp.ndarray  # (n_groups, n_groups, max_components, n_levels)
    n_modes: jnp.ndarray      # (n_groups, n_groups) — actual modes per patch


def get_gaussian_weights(config: DiscretiseConfig) -> jnp.ndarray:
    """Radial Gaussian weighting mask for a flattened patch (sigma = group_size)."""
    g = config.group_size
    x = jnp.arange(g) - (g - 1) / 2
    xx, yy = jnp.meshgrid(x, x)
    sigma = float(g)
    w2d = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return jnp.tile(w2d, (config.n_channels, 1, 1)).flatten()


def partition_into_patches(images: jnp.ndarray, config: DiscretiseConfig) -> jnp.ndarray:
    """Partition images into spatial patches.

    Args:
        images: (N, C, H, W) if n_channels > 1, or (N, H, W) if n_channels == 1
        config: discretisation parameters

    Returns:
        (N, n_groups, n_groups, C, g, g) patches
    """
    if images.ndim == 3:
        images = images[:, None, :, :]  # (N, H, W) -> (N, 1, H, W)
    N, C, _H, _W = images.shape
    g = config.group_size
    n = config.image_size // g
    return images.reshape(N, C, n, g, n, g).transpose(0, 2, 4, 1, 3, 5)


def compute_svd_basis(images: jnp.ndarray, config: DiscretiseConfig) -> SVDBasis:
    """
    Learn SVD basis and bin centres from structure images.

    Implements adaptive mode selection matching SPM's spm_svd:
    - Retains modes where S[i]/S[0] > 1/sv_threshold
    - Capped at max_components
    - Uses masking with fixed shapes for JAX compatibility

    Args:
        images: (N, H, W) or (N, C, H, W) preprocessed structure images
        config: discretisation parameters

    Returns:
        SVDBasis with learned projection and quantisation parameters
    """
    g = config.group_size
    n = config.image_size // g
    k = config.max_components
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
        _U, S_full, Vh = jnp.linalg.svd(data, full_matrices=False)
        V = Vh.T[:, :k]
        S = S_full[:k]

        # Adaptive mode selection: retain modes where S[i]/S[0] > 1/sv_threshold
        threshold = 1.0 / config.sv_threshold
        normalized = S / jnp.maximum(S[0], 1e-12)
        mode_mask = (normalized > threshold).astype(jnp.float32)  # (k,)
        n_modes = mode_mask.sum().astype(jnp.int32)

        # Zero out unused modes
        V = V * mode_mask[None, :]
        S = S * mode_mask

        return V, S, n_modes

    V_flat, S_flat, n_modes_flat = vmap(svd_one)(centred_2d)
    V = V_flat.reshape(n, n, f, k)
    S = S_flat.reshape(n, n, k)
    n_modes = n_modes_flat.reshape(n, n)

    # Project structure images to compute bin centres
    variates = jnp.einsum('bijf,ijfc->bijc', centred, V)  # (N, n, n, k)

    # Create a mode mask for zeroing out bin centres of unused modes
    mode_mask = (S > 0).astype(jnp.float32)  # (n, n, k)

    def symmetric_bin_centres(v):
        """Compute bin centres via linspace(-max_abs, max_abs, n_levels)."""
        max_abs = jnp.maximum(jnp.abs(v).max(), 1e-8)
        return jnp.linspace(-max_abs, max_abs, config.n_levels)

    # vmap over (n, n, k) -> bin_centres (n, n, k, n_levels)
    bin_centres = vmap(vmap(vmap(symmetric_bin_centres)))(variates.transpose(1, 2, 3, 0))

    # Zero out bin centres for unused modes
    bin_centres = bin_centres * mode_mask[:, :, :, None]

    return SVDBasis(V=V, S=S, mean=mean, bin_centres=bin_centres, n_modes=n_modes)


def encode_images(images: jnp.ndarray, basis: SVDBasis, config: DiscretiseConfig) -> jnp.ndarray:
    """
    Encode images as discrete observations using nearest-bin assignment.

    SPM uses: [~, U] = min(abs(u(t,m) - a)) where a is a vector of bin centres.

    Returns:
        (N, n_groups, n_groups, max_components) integer bin indices in [0, n_levels-1]
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

    # Nearest-bin assignment: argmin(|variate - centre|)
    # variates: (N, n, n, k), bin_centres: (n, n, k, n_levels)
    # Expand for broadcasting: (N, n, n, k, 1) vs (1, n, n, k, n_levels)
    diffs = jnp.abs(variates[..., None] - basis.bin_centres[None, :, :, :, :])
    indices = jnp.argmin(diffs, axis=-1)  # (N, n, n, k)

    return indices


def decode_observations(
    observations: jnp.ndarray, basis: SVDBasis, config: DiscretiseConfig
) -> jnp.ndarray:
    """
    Reconstruct images from discrete observations.

    Uses bin centres directly (no edge midpoint computation needed).

    Args:
        observations: (N, n_groups, n_groups, max_components) integer bin indices

    Returns:
        (N, C, H, W) reconstructed images
    """
    g = config.group_size
    n = config.image_size // g
    C = config.n_channels

    # Gather the centre for each observation
    def gather_centres(obs_ijn, centres_ijn):
        # obs_ijn: (k,) int indices, centres_ijn: (k, n_levels)
        return centres_ijn[jnp.arange(centres_ijn.shape[0]), obs_ijn]

    def gather_image(obs_i):
        # obs_i: (n, n, k)
        return vmap(vmap(gather_centres))(obs_i, basis.bin_centres)

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
