import cv2
import numpy as onp
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array
from typing import Tuple, List    
import itertools


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

def map_rbg_2_discrete(image_data: Array, tile_diameter=32, n_bins=9, max_n_modes=32, sv_thr=32, t_resampling=2):
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

    L = spm_combinations(shape_no_time)

    g, h, m = spm_tile(L, width=shape_no_time[1], height=shape_no_time[2], tile_diameter=tile_diameter)
    
    

def spm_combinations(shape):
    """
    Generate a matrix of all combinations of indices given a vector of dimensions.
    
    Parameters:
    shape (list or array): A vector of dimensions.
    
    Returns:
    jnp.ndarray: A matrix of all combinations of indices.
    """
    dim_ranges = [list(jnp.arange(i)) for i in shape] 
    return list(itertools.product(*dim_ranges))



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


def spm_tile(L: List, width: int, height: int, tile_diameter: int=32):
    """
    Grouping into a partition of non-overlapping outcome tiles
    This routine identifies overlapping groups of pixels, returning their
    mean locations and a cell array of weights (based upon radial Gaussian
    basis functions) for each group. In other words, the grouping is based
    upon the location of pixels; in the spirit of a receptive field afforded
    by a sensory epithelium.Effectively, this leverages the conditional
    independencies that inherit from local interactions; of the kind found in
    metric spaces that preclude action at a distance.

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

    # TODO: make this faster
    L_array = jnp.asarray(L)[:,1:] # don't care about the 0-th dimension of combinations

    h = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
    g = [[None for _ in range(n_columns)] for _ in range(n_rows)] 
    for i in range(n_rows): 
        for j in range(n_columns): 
            distance_evals = vmap(lambda x: distance(x, jnp.array([x[i], y[j]])))(L_array)

            ij = jnp.argwhere(distance_evals < 2 * tile_diameter) 
            h[i][j] = jnp.exp(-distance_evals / (2 * (tile_diameter / 2)**2))
            g[i][j] = ij

    g_flat = flatten(g)
    h_flat = flatten(h)

    num_groups = n_rows * n_columns
    
    # weighting of groups 
    h_matrix = jnp.stack(h_flat) # [num_groups, n_pixels_per_group)
    h = spm_dir_norm(h_matrix) # normalize across groups


    H_weights = [h[g,g_flat[g]] for g_i in range(num_groups)]

    M = jnp.zeros((num_groups, 2))
    for g in range(num_groups):
        M.at[g, :].set(jnp.mean(L[g_flat[g]], axis=0))
    
    return g_flat, M, H_weights


def simple_square_tile(x, patch_size):
    # x : image in [C, H, W]
    # patch_size = [h, w], i.e. [32,32]
    channels = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    x = x.reshape(channels, height // patch_size[0], patch_size[0], width // patch_size[1], patch_size[1])
    x = x.transpose(1, 3, 2, 4, 0)
    x = x.reshape(-1, *x.shape[2:])
    return x

if __name__ == "__main__":
    
    path_to_file = "examples/structure_learning/dove.mp4"

    # Read in the video file as tensor (num_frames, width, height, channels)
    frames = read_frames_from_mp4(path_to_file)

    # Map the RGB image to discrete outcomes
    map_rbg_2_discrete(frames)

        

    