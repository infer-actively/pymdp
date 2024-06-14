import cv2
import numpy as onp
from jax import numpy as jnp
from jax import vmap
from jax import nn
from jaxtyping import Array
from typing import Tuple, List    
import itertools

from math import prod

def read_frames_from_mp4(file_path: str, num_frames: int = 32, size: tuple[int] = (128, 128)):
    """" read frames from an mp4 file """
    cap = cv2.VideoCapture(file_path)

    width, height = size
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x_center = video_width // 2
    y_center = video_height // 2
    x_start = max(0, x_center - width // 2)
    y_start = max(0, y_center - height // 2)

    frame_indices = jnp.linspace(0, total_frames - 1, num_frames, dtype=int)

    # @TODO: Why did Karl concatenate a sequence of sub-sampled frames here (line 56 in DEM_compression.m)
    frame_indices = jnp.concatenate((frame_indices, frame_indices))
    frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx.item()))

        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y_start:y_start+height, x_start:x_start+width]
        resized_frame = cv2.resize(cropped_frame, size)
        frames.append(resized_frame)
    
    cap.release()
    return jnp.array(frames)

def map_rbg_2_discrete(image_data: Array, tile_diameter=32, n_bins=9, max_n_modes=32, sv_thr=(1./32.), t_resampling=2):
    """ Re-implementation of `spm_rgb2O.m` in Python
    Maps an RGB image format to discrete outcomes

    Args:
    image_data: Array
        The image data to be mapped to discrete outcomes. Shape (num_frames, width, height, channels)
    tile_diameter: int
        Diameter of the tiles (`nd` in the original code)
    n_bins: int
        Number of variates (`nb` in the original code)
    max_n_modes: int
        Maximum number of modes (`nm` in the original code)
    sv_thr: int
        Threshold for singular values (`su` in the original code)
    t_resampling: int
        Threshold for temporal resampling (`R` in the original code)
    """
    # ensure number of bins is odd (for symmetry)
    n_bins = 2 * jnp.trunc(n_bins/2) + 1

    # Permute for grouping
    n_frames = image_data.shape[0]

    T = int(t_resampling * jnp.trunc(n_frames/t_resampling)) # length of time partition

    # tranposes be channel, time, width, height
    image_data = jnp.transpose(image_data[:T,...], (3, 0, 1, 2)) # truncate the time series and transpose the axes to the right place

    # Get the shape of the data
    n_channels, _, width, height = image_data.shape

    # channel, time, width, height
    image_data = image_data.reshape((n_channels*t_resampling, -1, width, height))

    # move to sub-sampled-time, channel * 2, width, height
    image_data = jnp.transpose(image_data, (1, 0, 2, 3))

    # shape of the data excluding the time dimension
    shape_no_time = image_data.shape[1:]

    #L = spm_combinations(shape_no_time)

    # print(shape_no_time[0])

    patch_indices, patch_centroids, patch_weights = spm_tile(width=shape_no_time[1], height=shape_no_time[2], n_copies=shape_no_time[0], tile_diameter=tile_diameter)

    return patch_svd(image_data, patch_indices, patch_centroids, patch_weights, sv_thr), patch_indices
    
def patch_svd(image_data: Array, patch_indices: List[Array], patch_centroids, patch_weights: List[Array], sv_thr: float = 1e-6, max_n_modes: int = 32, n_bins: int=9):
    """
    image_data: [time, channel, width, height]
    patch_indices: [[indicies_for_patch] for num_patches]
    patch_weights: [[indices_for_patch, num_patches] for num_patches]
    """

    n_frames, channels_x_duplicates, width, height = image_data.shape
    o_idx = 0
    observations = []
    locations_matrix = []
    group_indices = [] 
    sv_discrete_axis = []
    V_per_patch = [None] * len(patch_indices)
    for g_i, patch_g_indices in enumerate(patch_indices):

        # single value decomposition for this 'pixel group''
        Y = image_data.reshape(n_frames, channels_x_duplicates*width*height)[:,patch_g_indices] * patch_weights[g_i]
        
        # (n_frames x n_frames), (n_frames,), (n_frames x n_frames)
        U, svals, V = jnp.linalg.svd(Y@Y.T, full_matrices=True)
        
        normalized_svals = svals * (len(svals)/svals.sum())
        topK_svals = (normalized_svals > sv_thr) # equivalent of `j` in spm_svd.m
        topK_s_vectors = U[:, topK_svals]

        projections = Y.T @ topK_s_vectors  # do equivalent of spm_en on this one
        projections_normed = projections / jnp.linalg.norm(projections, axis=0, keepdims=True)

        svals = jnp.sqrt(svals[topK_svals])

        num_modalities = min(len(svals),max_n_modes)

        if num_modalities > 0:
            V_per_patch[g_i] = projections_normed[:, :num_modalities]
            weighted_topk_s_vectors = topK_s_vectors[:, :num_modalities] * svals[:num_modalities]

        # generate (probability over discrete) outcomes
        for m in range(num_modalities):
            
            # dicretise singular variates
            d = jnp.max(jnp.abs(weighted_topk_s_vectors[:,m]))

            # this determines the nunber of bins
            projection_bins = jnp.linspace(-d, d, n_bins)

            observations.append([])
            for t in range(n_frames):
                
                # finds the index of of the projection at time t, for singular vector m, in the projection bins -- this will determine how it gets discretized
                min_indices = jnp.argmin(jnp.absolute(weighted_topk_s_vectors[t,m] - projection_bins))

                # observations are a one-hot vector reflecting the quantization of each singular variate into one of the projection bins
                observations[o_idx].append(nn.one_hot(min_indices, n_bins))
            
            # record locations and group for this outcome
            locations_matrix.append(patch_centroids[g_i,:])
            group_indices.append(g_i)
            sv_discrete_axis.append(projection_bins)
            o_idx += 1
    
    locations_matrix = jnp.stack(locations_matrix)

    return observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch


def map_discrete_2_rgb(observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch, patch_indices, image_shape):
    # observations list[list[array]] - list 0 is modalities list 1 is time arrays are onehot
    # image = jnp.zeros(image_shape)


    n_groups = len(patch_indices)

    recons_image = jnp.zeros(prod(image_shape))
    
    for group_idx in range(n_groups):

        modality_idx_in_patch = [modality_idx for modality_idx, g_i in enumerate(group_indices) if g_i == group_idx]
        num_modalities_in_patch = len(modality_idx_in_patch)    
        
        matched_bin_values = []
        for m in range(num_modalities_in_patch):
            m_idx = modality_idx_in_patch[m]
            matched_bin_values.append(sv_discrete_axis[m_idx].dot(observations[m_idx]))
        
        matched_bin_values = jnp.array(matched_bin_values)
        if len(matched_bin_values) > 0:
            recons_image = recons_image.at[patch_indices[group_idx]].set(recons_image[patch_indices[group_idx]] + V_per_patch[group_idx]*matched_bin_values)

    return recons_image.reshape(image_shape)

def spm_dir_norm(a):
    """
    Normalisation of a (Dirichlet) conditional probability matrix
    Args:
        A: (Dirichlet) parameters of a conditional probability matrix
    Returns:
        A: normalised conditional probability matrix
    """
    
    a0 = jnp.sum(a, axis=0)
    i = a0 > 0
    a = jnp.where(i, a / a0, a)
    a = a.at[:, ~i].set(1 / a.shape[0]) 
    return a    


def spm_tile(width: int, height: int, n_copies: int, tile_diameter: int=32):
    """
    Grouping into a partition of non-overlapping outcome tiles
    This routine identifies overlapping groups of pixels, returning their
    mean locations and a cell array of weights (based upon radial Gaussian
    basis functions) for each group. In other words, the grouping is based
    upon the location of pixels; in the spirit of a receptive field afforded
    by a sensory epithelium.Effectively, this leverages the conditional
    independencies that inherit from local interactions; of the kind found in
    metric spaces that preclude action at a distance.
    
    Args:
        L: list of indices
        width: width of the image
        height: height of the image
        tile_diameter: diameter of the tiles
    Returns:
        G: outcome indices
        M: (mean) outcome location
        H: outcome weights 
    """
    def distance(x, y): 
        return jnp.sqrt(((x - y) ** 2).sum())

    def flatten(l):
        return [item for sublist in l for item in sublist]

    # Centroid locations
    n_rows = int((width + 1)/ tile_diameter)
    n_columns = int((height + 1)/ tile_diameter)
    x = jnp.linspace(tile_diameter / 2 - 1, width - tile_diameter/2, n_rows)
    y = jnp.linspace(tile_diameter / 2 - 1, height - tile_diameter/2, n_columns)

    pixel_indices = n_copies * [jnp.array(jnp.meshgrid(jnp.arange(width), jnp.arange(height))).T.reshape(-1, 2)]
    pixel_indices = jnp.concatenate(pixel_indices, axis=0)

    h = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
    g = [[None for _ in range(n_columns)] for _ in range(n_rows)] 
    for i in range(n_rows):
        for j in range(n_columns):
            pos = jnp.array([x[i], y[j]])
            distance_evals = vmap(lambda x: distance(x, pos))(pixel_indices)

            ij = jnp.argwhere(distance_evals < 2 * tile_diameter).squeeze()
            h[i][j] = jnp.exp(-distance_evals / (2 * (tile_diameter / 2)**2))
            g[i][j] = ij

    G = flatten(g)
    h_flat = flatten(h)

    num_groups = n_rows * n_columns
    
    # weighting of groups 
    h_matrix = jnp.stack(h_flat) # [num_groups, n_pixels_per_group)
    h = spm_dir_norm(h_matrix) # normalize across groups

    H_weights = [h[g_i, G[g_i]] for g_i in range(num_groups)]

    M = jnp.zeros((num_groups, 2))
    for g_i in range(num_groups):

        M.at[g_i, :].set(pixel_indices[G[g_i]].mean(0))
    
    return G, M, H_weights


if __name__ == "__main__":
    
    path_to_file = "examples/structure_learning/dove.mp4"

    # Read in the video file as tensor (num_frames, width, height, channels)
    frames = read_frames_from_mp4(path_to_file)

    # Map the RGB image to discrete outcomes
    (observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch), patch_indices = map_rbg_2_discrete(frames)

    # Map the discrete outcomes back to RGB    
    observations = jnp.array(observations)
    video = jnp.zeros(frames.shape)
    for t in range(observations.shape[1]):
        video[t, ...] = map_discrete_2_rgb(observations[:, t, :], locations_matrix, group_indices, sv_discrete_axis, V_per_patch, patch_indices, frames.shape[1:])


    # write out the shapes of the observations
    print(f'Number of observations: {len(observations)}')
    print(f'Number of timesteps: {len(observations[0])}')
    print(f'Dimensionality of observations: {observations[0][0].shape}')

    print(f'Size of locations matrix: {locations_matrix.shape}')

    print(f'Number of group indices should equal number of modalities: {len(group_indices)}')
    print(f'Show the group indices: {group_indices}')

        

    