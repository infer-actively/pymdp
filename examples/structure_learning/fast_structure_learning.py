import cv2
import numpy as onp
from jax import numpy as jnp
from jax import vmap
from jax import nn
from jax import tree_util as jtu
from jaxtyping import Array
from typing import Tuple, List    
import itertools
import matplotlib.pyplot as plt
import imageio

from pymdp.jax.agent import Agent


from math import prod

def read_frames_from_npz(file_path: str, num_frames: int = 32, rollout: int = 0):
    """ read frames from a npz file from atari expert trajectories """
    # shape is [num_rollouts, num_frames, 1, height, width, channels]
    res = onp.load(file_path)
    frames = res['arr_0'][rollout, 0:num_frames, 0, ...]
    return frames

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
    frame_indices = jnp.concatenate((frame_indices, frame_indices), axis=0)
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

    n_frames, width, height, n_channels = image_data.shape
    T = int(t_resampling * jnp.trunc(n_frames/t_resampling)) # length of time partition

    # transpose to [T x C x W x H]
    image_data = jnp.transpose(image_data[:T,...], (0, 3, 1, 2)) # truncate the time series and transpose the axes to the right place

    # concat each t_resampling frames
    image_data = image_data.reshape((T//t_resampling, -1, width, height))

    # shape of the data excluding the time dimension ((t_resampling * C) x W x H)
    shape_no_time = image_data.shape[1:]

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
            
            # discretise singular variates
            d = jnp.max(jnp.abs(weighted_topk_s_vectors[:,m]))

            # this determines the number of bins
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


def map_discrete_2_rgb(observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch, patch_indices, image_shape, t_resampling=2):
    n_groups = len(patch_indices)

    # image_shape given as [W H C]
    shape = [t_resampling, image_shape[-1], image_shape[-3], image_shape[-2]]

    recons_image = jnp.zeros(prod(shape))
    
    for group_idx in range(n_groups):
        modality_idx_in_patch = [modality_idx for modality_idx, g_i in enumerate(group_indices) if g_i == group_idx]
        num_modalities_in_patch = len(modality_idx_in_patch)    
        
        matched_bin_values = []
        for m in range(num_modalities_in_patch):
            m_idx = modality_idx_in_patch[m]
            matched_bin_values.append(sv_discrete_axis[m_idx].dot(observations[m_idx]))
        
        matched_bin_values = jnp.array(matched_bin_values)
        if len(matched_bin_values) > 0:
            recons_image = recons_image.at[patch_indices[group_idx]].set(recons_image[patch_indices[group_idx]] + V_per_patch[group_idx].dot(matched_bin_values))

    recons_image = recons_image.reshape(shape)
    return recons_image

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
        M = M.at[g_i, :].set(pixel_indices[G[g_i],:].mean(0))
    
    return G, M, H_weights

def spm_space(L: Array):
    """ 
    This function takes a set of modalities and their
    spatial coordinates and decimates over space into a compressed 
    set of modalities, and assigns the previous modalities
    to the new set of modsalities. 

    Args:
        L (Array): num_modalities x 2
    Returns:
        G (List[Array[int]]): 
            outcome indices mapping new modalities indices to
            previous modality indices
    """

    # this is the second case (skipping if isvector(L))
    # locations
    Nl = L.shape[0]
    unique_locs = jnp.unique(L, axis=0)
    Ng = unique_locs.shape[0]
    Ng = jnp.ceil(jnp.sqrt(Ng/4))
    if Ng == 1:
        G = list(range(Nl))
        return G

    # decimate locations
    x = jnp.linspace(jnp.min(L[:,0]), jnp.max(L[:,0]), int(Ng))
    y = jnp.linspace(jnp.min(L[:,1]), jnp.max(L[:,1]), int(Ng))
    R = jnp.fliplr(jnp.array(jnp.meshgrid(x, y)).T.reshape(-1, 2))
    
    # nearest (reduced) location
    closest_loc = lambda loc: jnp.argmin(jnp.linalg.norm(R - loc, axis=1))
    g = vmap(closest_loc)(L)

    # grouping partition
    G = []

    # these two lines do the equivalent of u = unique(g, 'stable') in MATLAB
    _, unique_idx = jnp.unique(g, return_index=True)
    u = g[jnp.sort(unique_idx)]
    for i in range(len(u)):
        G.append(jnp.argwhere(g == u[i]).squeeze())

    return G

def spm_time(T, d):
    """
    Grouping into a partition of non-overlapping sequences
    Args:
    T (int): total number of the timesteps
    d (int): number timesteps per partition

    Returns:
    list: A list of partitions with non-overlapping sequences
    """
    t = []
    for i in range(T // d):
        t.append(jnp.arange(d) + (i * d))
    return t

def spm_unique(a):
    """
    Fast approximation by simply identifying unique locations in a
    multinomial statistical manifold, after discretising to probabilities of
    zero, half and one (using Matlab’s unique and fix operators).
    
    Args:
        a: array (n, x)
    Returns:
        sorted indices of unique x'es
    """

    
    # Discretize to probabilities of zero, half, and one
    # 0 to 0.5 -> 0, 0.5 to 1 -> 1, 1 -> 2
    o_discretized = jnp.fix(2 * a)
    
    # Find unique rows
    _, j = jnp.unique(o_discretized, return_inverse=True, axis=0)
    
    return jnp.sort(j).squeeze(axis=1)

def spm_structure_fast(observations, dt=2):
    """
    Args:
        observations (array): (num_modalities, num_steps, num_obs)
        dt (int)
    """

    # Find unique outputs per timestep
    num_modalities, num_steps, num_obs = observations.shape
    o = jnp.moveaxis(observations,1,0).reshape(num_steps, -1)
    j = spm_unique(o)

    # Likelihood tensors
    Ns = len(jnp.unique(j))  # number of latent causes

    a = num_modalities*[None]

    for m in range(num_modalities):
        a[m] = jnp.zeros((num_obs, Ns))
        for s in range(Ns):
            a[m] = a[m].at[:,s].set(observations[m,j == s].mean(axis=0)) # observations[m,j == s] will have shape (num_timesteps_that_match, num_bins)

    # Transition tensors
    if dt < 2:
        # no dynamics
        b = [jnp.eye(Ns)]
        return a, b
    
    # Assign unique transitions between states to paths
    b = jnp.zeros((Ns, Ns, 1))

    for t in range(len(j)-1):
        if not jnp.any(b[j[t+1], j[t], :]):
            # does this state have any transitions under any paths
            u = jnp.where(~jnp.any(b[:, j[t], :], axis=0))[0]
            if len(u) == 0:
                # Add new path if no empty paths found
                b = jnp.concatenate((b, jnp.zeros((Ns, Ns, 1))), axis=2)
                b = b.at[j[t + 1], j[t], -1].set(1)
            else:
                # Use first empty path
                b = b.at[j[t + 1], j[t], u].set(1)
    
    return a, b


def spm_MB_structure_learning(observations, locations_matrix, dt: int = 2, max_levels: int = 8):
    """

    Args:
        observations (array): (num_modalities, time, num_obs)
        locations_matrix (array): (num_modalities, 2) 
    """

    A, B, RG, LG, = [], [], [], []
    observations = [observations]
    for n in range(max_levels):
        G = spm_space(locations_matrix)
        T = spm_time(observations[n].shape[1], dt)

        A, B = [], []
        A_dependencies = []
        for g in range(len(G)):
            a, b = spm_structure_fast(observations[n][G[g]], dt)
            A += a
            B += b

            # dependencies
            A_deps_for_patch_g = [g] * len(G[g])
            A_dependencies += A_deps_for_patch_g
  
        A_dependencies = [[a_dep] for a_dep in A_dependencies]
        
        RG.append(G)
        LG.append(locations_matrix)

        pdp = Agent(A=A, B=B, A_dependencies=A_dependencies, apply_batch=True, onehot_obs=True)

        # Solve at the next timescale
        for t in range(len(T)):
            

            sub_horizon = len(T[t])
            for j in range(sub_horizon):
                
                current_obs = []
                for g in range(len(G)):
                    for m_g in range(len(G[g])):
                        # get a new observation from the n-th hierarchical level, the G[g] different modality indices for this patch, the T[t][j]-th timestep.
                        # the [None,...] is to add a trivial batch dimension
                        obs_n_g_t = observations[n][G[g][m_g],T[t][j],:][None,...]
                        current_obs.append(obs_n_g_t)

                # if we're beyond the first timestep, append observations_list to a growing list of historical observations
                if j > 0:
                    observations_list = jtu.tree_map(
                    lambda prev_o, new_o: jnp.concatenate([prev_o, jnp.expand_dims(new_o, 1)], 1), previous_obs, current_obs
                    )

                if j == 0:
                    qs, _ = pdp.infer_states(observations_list, past_actions=None, empirical_prior=pdp.D)
                else:
                    qs, _ = pdp.infer_states(observations_list, past_actions=None, empirical_prior=empirical_prior)

                # fix to only one path for now?
                # how do we get the actual path? infer? plan?
                action = jnp.zeros((1))
                empirical_prior, _ = pdp.infer_empirical_prior(action, qs)
                
                previous_obs = jtu.tree_map(lambda x: jnp.copy(x), observations_list) # set the current observation (and history) equal to the previous set of observations

               
            
            pdp   = MDP{n};
            pdp.T = numel(T{t}); 
            for j = 1:pdp.T
                pdp.O(:,j) = O{n}(:,T{t}(j));
            end
            pdp   = spm_MDP_VB_XXX(pdp);

            % initial states and paths
            %------------------------------------------------------------------
            ig    = 1;
            L     = zeros(1,size(L,2));
            for g = 1:numel(G)

                % states, paths and average location for this goup
                %--------------------------------------------------------------
                qs = pdp.X{g}(:,1);
                qu = pdp.P{g}(:,end);
                ml = mean(MDP{n}.LG(G{g},:));

                % states (odd)
                %--------------------------------------------------------------
                MDP{n}.id.D{g} = ig;
                O{n + 1}{ig,t} = qs;
                L(ig,:)        = ml;
                ig = ig + 1;

                % paths (even)
                %--------------------------------------------------------------
                MDP{n}.id.E{g} = ig;
                O{n + 1}{ig,t} = qu;
                L(ig,:)        = ml;
                ig = ig + 1;

            end
        end

        

    return A, B

if __name__ == "__main__":
    
    path_to_file = "examples/structure_learning/dove.mp4"

    # Read in the video file as tensor (num_frames, width, height, channels)
    frames = read_frames_from_mp4(path_to_file)

    # Map the RGB image to discrete outcomes
    # Observations are list[list[array]] -> num modalities, time-steps, num_discrete_bins
    # Location matrix is num_modalities x 2 (width, height)
    # Group indices is num_modalities
    # sv_discrete_axis num_modalities x num_discrete_bins
    # V_per_patch num_patches, num_pixels_per_patch x 11?
    (observations, locations_matrix, group_indices, sv_discrete_axis, V_per_patch), patch_indices = map_rbg_2_discrete(frames, tile_diameter=32, n_bins=16)

    # convert list of list of observation one-hots into an array of size (num_modalities, timesteps, num_obs)
    observations = jnp.asarray(observations)
    
    # Run structure learning on the observations
    A, B = spm_MB_structure_learning(observations, locations_matrix, max_levels=1)
    # for A_m in A[0]:
    #     print(A_m[0].shape)
    #     print(A_m)
    # for B_m in B[0]:
    #     print(B_m[0].shape)
    #     print(B_m)

    # G = spm_space(locations_matrix)
    # print(G)

    
    # ims = []

    # # Map the discrete outcomes back to RGB    
    # observations = jnp.array(observations)
    # video = jnp.zeros(frames.shape)
    # for t in range(observations.shape[1]):
    #     img = map_discrete_2_rgb(observations[:, t, :], locations_matrix, group_indices, sv_discrete_axis, V_per_patch, patch_indices, frames.shape[-3:])

    #     # this reconstructs 2 frames
    #     for i in range(2):
    #         im = img[i, ...]
    #         # transform back to RGB
    #         im = jnp.transpose(im, (1, 2, 0))
    #         im /= 255
    #         im = jnp.clip(im, 0, 1)
    #         im = (255*im).astype(onp.uint8)

    #         gt = frames[t*2 + i]
    #         ims.append(onp.hstack([im, gt]) )

    # imageio.mimsave('reconstruction.gif', ims)



    