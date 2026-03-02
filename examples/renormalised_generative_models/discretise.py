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

import numpy as np
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
    config: DiscretiseConfig = DiscretiseConfig()


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

        # Adaptive mode selection matching spm_svd:
        # eigenvalue_i * n / sum(eigenvalues) > 1/su
        threshold = 1.0 / config.sv_threshold
        eigvals = jnp.square(S)
        L = eigvals.shape[0]
        normalized = eigvals * (L / jnp.maximum(eigvals.sum(), 1e-12))
        mode_mask = (normalized > threshold).astype(jnp.float32)

        # Keep at least one mode for near-degenerate patches
        mode_mask = mode_mask.at[0].set(
            jnp.where(mode_mask.sum() > 0, mode_mask[0], 1.0)
        )
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

    return SVDBasis(V=V, S=S, mean=mean, bin_centres=bin_centres, n_modes=n_modes, config=config)


def encode_images(images: jnp.ndarray, basis: SVDBasis) -> jnp.ndarray:
    """
    Encode images as discrete observations using nearest-bin assignment.

    SPM uses: [~, U] = min(abs(u(t,m) - a)) where a is a vector of bin centres.

    Returns:
        (N, n_groups, n_groups, max_components) integer bin indices in [0, n_levels-1]
    """
    g = basis.config.group_size
    n = basis.config.image_size // g
    f = basis.config.n_channels * g * g

    patches = partition_into_patches(images, basis.config)
    N = patches.shape[0]
    flat = patches.reshape(N, n, n, f)

    weights = get_gaussian_weights(basis.config)
    centred = flat * weights[None, None, None, :] - basis.mean[None, :, :, :]

    variates = jnp.einsum('bijf,ijfc->bijc', centred, basis.V)

    # Nearest-bin assignment: argmin(|variate - centre|)
    # variates: (N, n, n, k), bin_centres: (n, n, k, n_levels)
    # Expand for broadcasting: (N, n, n, k, 1) vs (1, n, n, k, n_levels)
    diffs = jnp.abs(variates[..., None] - basis.bin_centres[None, :, :, :, :])
    indices = jnp.argmin(diffs, axis=-1)  # (N, n, n, k)

    return indices


def decode_observations(
    observations: jnp.ndarray, basis: SVDBasis
) -> jnp.ndarray:
    """
    Reconstruct images from discrete observations.

    Uses bin centres directly (no edge midpoint computation needed).

    Args:
        observations: (N, n_groups, n_groups, max_components) integer bin indices

    Returns:
        (N, C, H, W) reconstructed images
    """
    g = basis.config.group_size
    n = basis.config.image_size // g
    C = basis.config.n_channels

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
    weights = get_gaussian_weights(basis.config)
    recon = recon / weights[None, None, None, :]

    # Reassemble: (N, n, n, C*g*g) -> (N, n, n, C, g, g) -> (N, C, H, W)
    patches = recon.reshape(-1, n, n, C, g, g)
    return patches.transpose(0, 3, 1, 4, 2, 5).reshape(-1, C, n * g, n * g)


# ---------------------------------------------------------------------------
# Overlapping tiles (SPM spm_tile strategy)
# ---------------------------------------------------------------------------


class OverlappingSVDBasis(NamedTuple):
    """SVD basis learned with SPM-style overlapping Gaussian-weighted tiles.

    Each group's receptive field extends to radius 2*d from its centroid,
    with Gaussian weighting (sigma=d/2) normalized across groups so each
    pixel's weights sum to 1 (spm_dir_norm).
    """
    tile_weights: jnp.ndarray  # (n_groups, n_groups, n_pixels) normalized
    V: jnp.ndarray             # (n_groups, n_groups, n_pixels, max_components)
    S: jnp.ndarray             # (n_groups, n_groups, max_components)
    mean: jnp.ndarray          # (n_groups, n_groups, n_pixels)
    bin_centres: jnp.ndarray   # (n_groups, n_groups, max_components, n_levels)
    n_modes: jnp.ndarray       # (n_groups, n_groups) actual modes per tile
    config: DiscretiseConfig = DiscretiseConfig()


def compute_tile_weights(config: DiscretiseConfig) -> jnp.ndarray:
    """Compute SPM-style overlapping tile weights matching spm_tile.

    Centroids are placed via linspace (matching SPM). Each tile includes
    pixels within radius 2*d, weighted by a Gaussian with sigma=d/2.
    Weights are normalized so each pixel's total weight across all groups = 1.

    Args:
        config: discretisation parameters

    Returns:
        (n_groups, n_groups, H*W*C) normalized weight map
    """
    d = config.group_size
    H = W = config.image_size
    C = config.n_channels
    ng = round(H / d)

    # Centroid locations matching SPM's linspace with 0-indexed coords
    # SPM (1-indexed): linspace(min + d/2 - 1, max - d/2, n)
    # 0-indexed: linspace(d/2 - 1, H - 1 - d/2, n)
    cx = np.linspace(d / 2 - 1, H - 1 - d / 2, ng)
    cy = np.linspace(d / 2 - 1, W - 1 - d / 2, ng)

    # Pixel coordinates
    pr, pc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H, W)
    pr_flat = pr.flatten().astype(np.float64)  # (H*W,)
    pc_flat = pc.flatten().astype(np.float64)

    sigma = d / 2.0
    radius = 2.0 * d

    # Compute weights: (ng, ng, H*W)
    raw = np.zeros((ng, ng, H * W), dtype=np.float64)
    for i in range(ng):
        for j in range(ng):
            dist_sq = (pr_flat - cx[i]) ** 2 + (pc_flat - cy[j]) ** 2
            gauss = np.exp(-dist_sq / (2 * sigma ** 2))
            gauss[np.sqrt(dist_sq) >= radius] = 0.0
            raw[i, j] = gauss

    # Normalize: each pixel's weights sum to 1 across all groups (spm_dir_norm)
    total = raw.sum(axis=(0, 1), keepdims=True)
    weights = raw / np.maximum(total, 1e-12)

    # Tile across channels if multi-channel
    if C > 1:
        weights = np.tile(weights, (1, 1, C))

    return jnp.array(weights.astype(np.float32))


def compute_svd_basis_overlapping(
    images: jnp.ndarray, config: DiscretiseConfig
) -> OverlappingSVDBasis:
    """Learn SVD basis using SPM-style overlapping tiles.

    Each tile's SVD operates on Gaussian-weighted pixels within radius 2*d
    of the tile centroid, matching spm_rgb2O / spm_tile.

    Args:
        images: (N, H, W) or (N, C, H, W) preprocessed structure images
        config: discretisation parameters

    Returns:
        OverlappingSVDBasis with learned projection and quantisation parameters
    """
    if images.ndim == 3:
        images = images[:, None, :, :]  # (N, H, W) -> (N, 1, H, W)
    N, C, H, W = images.shape
    ng = round(H / config.group_size)
    k = config.max_components
    n_pixels = C * H * W

    # Flatten images to (N, C*H*W) with channel-last pixel ordering
    # SPM uses (color, row, col) ordering — we match with (C, H, W) -> flatten
    flat = images.reshape(N, n_pixels)

    # Tile weights: (ng, ng, n_pixels)
    tile_weights = compute_tile_weights(config)

    # Weight each image by each tile: (ng*ng, N, n_pixels)
    # Reshape for vmap: (ng*ng, n_pixels) weights, broadcast over N
    tw_flat = tile_weights.reshape(ng * ng, n_pixels)
    weighted = flat[None, :, :] * tw_flat[:, None, :]  # (ng*ng, N, n_pixels)

    mean = weighted.mean(axis=1)  # (ng*ng, n_pixels)
    centred = weighted - mean[:, None, :]  # (ng*ng, N, n_pixels)

    # SVD returns at most min(N, n_pixels) components; pad to k
    max_rank = min(N, n_pixels)

    def svd_one(data):
        _U, S_full, Vh = jnp.linalg.svd(data, full_matrices=False)
        # Vh: (max_rank, n_pixels), pad to (k, n_pixels) if max_rank < k
        V_raw = Vh.T  # (n_pixels, max_rank)
        V = jnp.zeros((n_pixels, k))
        nk = min(max_rank, k)
        V = V.at[:, :nk].set(V_raw[:, :nk])
        S = jnp.zeros(k)
        S = S.at[:nk].set(S_full[:nk])

        # Adaptive mode selection matching spm_svd:
        # eigenvalue_i * n / sum(eigenvalues) > 1/su
        threshold = 1.0 / config.sv_threshold
        eigvals = jnp.square(S)
        L = eigvals.shape[0]
        normalized = eigvals * (L / jnp.maximum(eigvals.sum(), 1e-12))
        mode_mask = (normalized > threshold).astype(jnp.float32)

        # Keep at least one mode for near-degenerate patches
        mode_mask = mode_mask.at[0].set(
            jnp.where(mode_mask.sum() > 0, mode_mask[0], 1.0)
        )
        n_modes = mode_mask.sum().astype(jnp.int32)

        V = V * mode_mask[None, :]
        S = S * mode_mask

        return V, S, n_modes

    V_flat, S_flat, n_modes_flat = vmap(svd_one)(centred)
    # V_flat: (ng*ng, n_pixels, k), S_flat: (ng*ng, k)
    V = V_flat.reshape(ng, ng, n_pixels, k)
    S = S_flat.reshape(ng, ng, k)
    n_modes = n_modes_flat.reshape(ng, ng)
    mean = mean.reshape(ng, ng, n_pixels)

    # Project to compute bin centres
    # variates: for each group, project centred data onto V
    # centred: (ng*ng, N, n_pixels), V_flat: (ng*ng, n_pixels, k)
    variates_flat = jnp.einsum('gnf,gfk->gnk', centred, V_flat)  # (ng*ng, N, k)
    variates = variates_flat.reshape(ng, ng, N, k)

    mode_mask = (S > 0).astype(jnp.float32)

    def symmetric_bin_centres(v):
        max_abs = jnp.maximum(jnp.abs(v).max(), 1e-8)
        return jnp.linspace(-max_abs, max_abs, config.n_levels)

    # vmap over (ng, ng, k) -> bin_centres (ng, ng, k, n_levels)
    bin_centres = vmap(vmap(vmap(symmetric_bin_centres)))(
        variates.transpose(0, 1, 3, 2)  # (ng, ng, k, N)
    )
    bin_centres = bin_centres * mode_mask[:, :, :, None]

    return OverlappingSVDBasis(
        tile_weights=tile_weights,
        V=V, S=S, mean=mean,
        bin_centres=bin_centres, n_modes=n_modes,
        config=config,
    )


def encode_images_overlapping(
    images: jnp.ndarray,
    basis: OverlappingSVDBasis,
) -> jnp.ndarray:
    """Encode images using overlapping tile basis via nearest-bin assignment.

    Returns:
        (N, n_groups, n_groups, max_components) integer bin indices
    """
    if images.ndim == 3:
        images = images[:, None, :, :]
    N, C, H, W = images.shape
    n_pixels = C * H * W

    flat = images.reshape(N, n_pixels)

    # Weight and center: (ng, ng, N, n_pixels)
    tw = basis.tile_weights  # (ng, ng, n_pixels)
    weighted = flat[None, None, :, :] * tw[:, :, None, :]
    centred = weighted - basis.mean[:, :, None, :]

    # Project: (ng, ng, N, k)
    variates = jnp.einsum('ijnf,ijfk->ijnk', centred, basis.V)

    # Nearest-bin: (ng, ng, N, k, 1) vs (ng, ng, 1, k, n_levels)
    diffs = jnp.abs(
        variates[..., None] - basis.bin_centres[:, :, None, :, :]
    )
    indices = jnp.argmin(diffs, axis=-1)  # (ng, ng, N, k)

    return indices.transpose(2, 0, 1, 3)  # (N, ng, ng, k)


def decode_observations_overlapping(
    observations: jnp.ndarray,
    basis: OverlappingSVDBasis,
) -> jnp.ndarray:
    """Reconstruct images from discrete observations using overlapping tiles.

    Reconstruction sums the weighted contributions from all overlapping groups.
    Since tile weights are normalized (sum to 1 per pixel), the sum of weighted
    reconstructions recovers the original pixel values.

    Args:
        observations: (N, n_groups, n_groups, max_components) integer bin indices

    Returns:
        (N, C, H, W) reconstructed images
    """
    C = basis.config.n_channels
    H = W = basis.config.image_size
    n_pixels = C * H * W

    # Gather bin centres for each observation
    def gather_centres(obs_ijn, centres_ijn):
        return centres_ijn[jnp.arange(centres_ijn.shape[0]), obs_ijn]

    def gather_group(obs_ij, centres_ij):
        # obs_ij: (N, k), centres_ij: (k, n_levels)
        return vmap(lambda o: gather_centres(o, centres_ij))(obs_ij)

    # obs transposed to (ng, ng, N, k) for per-group processing
    obs_t = observations.transpose(1, 2, 0, 3)

    # For each group, reconstruct weighted pixel values and sum
    # Accumulate into (N, n_pixels)
    N = observations.shape[0]
    recon = jnp.zeros((N, n_pixels))

    # Vectorized: compute all groups at once
    # variates: (ng, ng, N, k)
    variates = vmap(
        vmap(gather_group, in_axes=(0, 0)),
        in_axes=(0, 0),
    )(obs_t, basis.bin_centres)  # (ng, ng, N, k)

    # Unproject each group: (ng, ng, N, n_pixels)
    recon_weighted = jnp.einsum('ijnk,ijfk->ijnf', variates, basis.V)
    recon_weighted = recon_weighted + basis.mean[:, :, None, :]

    # Sum across groups: since tile weights are normalized, summing the
    # weighted reconstructions recovers the pixel values
    recon = recon_weighted.sum(axis=(0, 1))  # (N, n_pixels)

    return recon.reshape(N, C, H, W)
