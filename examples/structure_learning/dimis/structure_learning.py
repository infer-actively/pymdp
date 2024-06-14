import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
import jax.random as jr

from numpy import prod
from jax import nn, vmap, lax
from utils import Patch

def over_all_patches(patched_images, batch_time_shape):
    
    shape = patched_images.shape
    num_patches = shape[1]
    images = patched_images.reshape((-1,) + shape[2:])
    a, index, counts = jnp.unique(images, axis=0, return_index=True, return_counts=True)
    sorted_idx = index.argsort()
    a = a[sorted_idx]
    num_states = len(index)

    image_sequence = patched_images.reshape(batch_time_shape + (num_patches, 1, -1))
    state_sequence = jnp.all(image_sequence == a.reshape(num_states, -1), -1).argmax(-1)

    A = [a] * num_patches
    num_states = [num_states] * num_patches
    state_sequence = jnp.moveaxis(state_sequence, -1, 0)
    pA = [counts[sorted_idx]] * num_patches

    return A, pA, num_states, state_sequence

def per_patch(patched_images, batch_time_shape):
    A = []
    # pA = []
    num_states = []
    state_sequence = []
    num_patches = patched_images.shape[1]

    for g in range(num_patches):
        img_patch = patched_images[:, g]
        # a, index, counts = jnp.unique(img_patch, axis=0, return_index=True, return_counts=True)
        a, index = jnp.unique(img_patch, axis=0, return_index=True)
        sorted_idx = index.argsort()

        # add unique state-likelihoods given the order of occurence in a time series
        A.append( a[sorted_idx] )
        # pA.append( counts[sorted_idx] )
        num_states.append( len(index) )

        img_sequence = img_patch.reshape( batch_time_shape + (-1,) )
        a = A[-1].reshape(num_states[-1], -1)
        state_sequence.append( jnp.all(jnp.expand_dims(img_sequence, -2) == a, -1).argmax(-1) )

    # return A, pA, num_states, jnp.stack(state_sequence, 0)
    return A, num_states, jnp.stack(state_sequence, 0)

def make_As(patched_images, batch_time_shape, share_As=False):
    if share_As:
        return over_all_patches(patched_images, batch_time_shape)
    else:
        return per_patch(patched_images, batch_time_shape)

def make_Bs(state_sequences, num_states,  noise_baseline=1.):
    transition_sequences = jnp.stack([state_sequences[..., :-1], state_sequences[..., 1:]], -1)
    all_transitions = transition_sequences.reshape(state_sequences.shape[0], -1, 2)

    pBs = []
    control_sequence = []
    num_paths = []
    num_patches = all_transitions.shape[0]
    for g in range(num_patches):
        u_trans = jnp.unique(all_transitions[g], axis=0)

        def step_fn(carry, xs):
            B = carry
            n_paths = B.shape[-1]

            ps, ns = xs

            b = B[:, ps]
            t = jnp.any(b, 0)
            _ns = b.argmax(0) * t - (1 - t)
            exists = _ns == ns

            e = jnp.any(exists)
            u = e * (n_paths - jnp.argmax(exists[::-1]) - 1) 
            u += (1 - e) * (1 - t).argmax()

            ps = jnp.expand_dims(nn.one_hot(ps, num_states[g]), -2)
            ns = jnp.expand_dims(nn.one_hot(ns, num_states[g]), -1)
            B += jnp.expand_dims(ps * ns, -1) * jnp.expand_dims(nn.one_hot(u, n_paths), (-2, -3))
            return B, u
        
        n_paths = jnp.max( jnp.sum(jnp.unique(u_trans[:, 0]) == u_trans[:, 0, None], 0) )
        B0 = jnp.zeros((num_states[g], num_states[g], n_paths))
        prev_states = transition_sequences[g, ..., 0]
        next_states = transition_sequences[g, ..., 1]

        func = lambda x, y: lax.scan(step_fn, B0, (x, y))
        batched_pB, controls = vmap(func)(prev_states, next_states)
        pB = batched_pB.sum(0)
        

        # fill empty transition with random observed or unobserved transitions
        alpha = noise_baseline / num_states[g] # weak prior
        all_existing_transitions = jnp.any(pB, -1)
        S = jnp.expand_dims((1 - all_existing_transitions) * alpha + (1-alpha) * jnp.eye(num_states[g]), -1)
        pB = jnp.where(jnp.any(pB, 0), pB + alpha, S)
        pBs.append(  pB  )
        num_paths.append(pB.shape[-1])

        control_sequence.append(controls)

    return pBs, num_paths, jnp.stack(control_sequence, -1)

def fast_structure_learning(batched_image_sequence, *, image_size, patch_size, in_chans=2, rewards=None, noise_baseline=1.):
    make_patches = Patch(image_size, patch_size, in_chans=in_chans, flatten=False)

    As = {}
    Bs = {}
    pAs = {}
    pBs = {}
    num_states = {}
    num_paths = {}
    max_state_paths = {}
    patchers = {}

    batch_time_shape = batched_image_sequence.shape[:2]
    _images = batched_image_sequence.reshape((-1, in_chans,) + image_size)
    obs = vmap(make_patches)(_images)
    level = 0
    max_state_paths[level] = [int(obs[..., 0].max().item()) + 1, int(obs[..., 1].max().item()) + 1]
    while True:
        print(level)

        # make states and likelihoods
        A, level_num_states, state_sequences = make_As(obs, batch_time_shape)

        # make controls and transition matrices
        if batch_time_shape[-1] < 2:
            pB = [jnp.ones((ns, ns, 1)) for ns in level_num_states]
            level_num_paths = [1 for ns in level_num_states]
        else:
            pB, level_num_paths, path_sequences = make_Bs(state_sequences, tuple(level_num_states), noise_baseline=noise_baseline)

        # update matrices for this level
        max_ns = max(level_num_states)
        A = jtu.tree_map(
            lambda x, ns: jnp.pad(x, ((0, max_ns-ns), (0, 0), (0, 0), (0, 0))), A, level_num_states
        )

        As[level] = jnp.stack(A)
        print("A", As[level].shape)

        max_ps = max(level_num_paths)
        Bs[level] = jnp.stack(
            jtu.tree_map(lambda x, ns, np: jnp.pad(x / x.sum(0), ((0, max_ns - ns), (0, max_ns-ns), (0, max_ps-np))), pB, level_num_states, level_num_paths)
        )
        print("B", Bs[level].shape)

        num_states[level] = level_num_states
        num_paths[level] = level_num_paths
        patchers[level] = make_patches

        #update level
        level += 1
        max_state_paths[level] = [max_ns, max_ps]

        if prod(make_patches.grid_size) == 1 or batch_time_shape[-1] < 2:
            # break if no more patches can be created or no mulitple time steps remaining
            break
        
        state_sequences = jnp.moveaxis(state_sequences, 0, -1)
        new_image_size = tuple(make_patches.grid_size)
        state_sequences = state_sequences.reshape(state_sequences.shape[:-1] + new_image_size)
        path_sequences = path_sequences.reshape(path_sequences.shape[:-1] + new_image_size)
        
        # make new bathed image sequence out of state and control sequences
        _images = jnp.stack([state_sequences[:, :-1], path_sequences], -3)[:, ::2]

        # patch image to create observations per patch for the next level
        make_patches = Patch(new_image_size, patch_size, in_chans=2, flatten=False)
        obs = vmap(make_patches)(_images.reshape( (-1,) + _images.shape[-3:]))

        # record new batch x time shape
        batch_time_shape = _images.shape[:2]

    if rewards is not None:
        T = batched_image_sequence.shape[1]
        r = rewards[:, :T]
        for _ in range(level - 1):
            r = r.reshape(r.shape[0], -1, 2).sum(-1)
        
        # v = r[r != 0]
        # states = state_sequences[..., r[None] != 0]
        states = state_sequences[..., r[None] > 0]
        if len(states) > 0:
            states = nn.one_hot(states, level_num_states[0])
            H = jnp.zeros((1, level_num_states[0])) + jnp.sum(states, 0)
        else:
            H = None

    else:
        H = None

    return (As, Bs, H, patchers, num_states, num_paths, max_state_paths)

batch_mult = vmap(lambda x, y: x @ y)
def forward_backward_algo(agent, obs, masks, D, E):
    # forward pass
    def step_fn(carry, xs):
        o, mask = xs
        agent, prior = carry
        beliefs = agent.infer_states(o, None, [prior], None, mask=mask)
        q_pi, _ = agent.infer_policies(beliefs)
        qs = beliefs[0][:, 0]
        pred = batch_mult(batch_mult(agent.B[0], q_pi), qs)

        return (agent, pred), (qs, q_pi, pred)
    
    prior = D[0]
    O = [obs[..., i, None, :] for i in range(obs.shape[-2])]
    M = [masks[..., i] for i in range(masks.shape[-1])] if masks is not None else masks
    agent = eqx.tree_at(lambda x: (x.E,), agent, (E[0],))
    (agent, _), sequence = lax.scan(step_fn, (agent, prior), (O, M))
    
    # backward pass
    def step_fn(carry, xs):
        qs, q_pi, qss = xs
        posterior, agent = carry

        marginal = jnp.einsum(
            'ijkl,ik,il,ij->ikl', agent.B[0], qs, q_pi, posterior/(qss + 1e-16)
        )
        
        posterior = marginal.sum(-1)
        q_pi = marginal.sum(-2)
        return posterior, q_pi
    
    # sequences will always be of lenght two
    qs = sequence[0][0]
    q_pi = sequence[1][0]
    qss = sequence[2][0]
    marginals = step_fn((sequence[0][-1], agent), (qs, q_pi, qss))
    qs = jnp.concatenate([marginals[0][None], sequence[0][-1:]], 0)
    q_pi = marginals[1]

    return qs, q_pi