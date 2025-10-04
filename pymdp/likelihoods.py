import numpyro.distributions as dist
from jax import lax
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan

def evolve_trials(agent, data):
    """
    Evolve agent beliefs through a sequence of trials.

    Parameters
    ----------
    agent: Agent
        Active inference agent
    data: dict
        Dictionary containing trial data from rollout. Must have keys:
        - 'observation': list of arrays with shape (batch, timesteps, 1) per modality
        - 'action': array with shape (batch, timesteps, num_controls)

    Returns
    -------
    tuple
        (probs, observations, actions) where probs are action probabilities

    Note
    ----
    This function automatically transposes rollout format data from
    (batch, timesteps, ...) to (timesteps, batch, ...) for processing.

    Example
    -------
    # Using rollout data directly
    last, info, env = rollout(agent, env, num_timesteps, rng_key)
    probs, obs, acts = evolve_trials(agent, info)
    """

    observations = data['observation']
    actions = data['action']

    # Transpose rollout format: (batch, time, ...) -> (time, batch, ...)
    # Rollout outputs observations where each array has shape (batch_size, num_timesteps, 1)
    # Prepare data for scan
    data_transposed = {
        'observation': [o.swapaxes(0, 1) for o in observations],
        'action': actions.swapaxes(0, 1),
    }

    def step_fn(carry, xs):
        empirical_prior = carry
        obs = xs['observation']
        acts = xs['action']

        qs = agent.infer_states(obs, empirical_prior)
        q_pi, _ = agent.infer_policies(qs)
        probs = agent.multiaction_probabilities(q_pi)

        empirical_prior, _ = agent.update_empirical_prior(acts, qs)
        return empirical_prior, (probs, obs, acts)

    prior = agent.D
    _, res = lax.scan(step_fn, prior, data_transposed)

    return res

def aif_likelihood(Nb, Nt, Na, data, agent):
    # Na -> batch dimension - number of different subjects/agents
    # Nb -> number of experimental blocks
    # Nt -> number of trials within each block

    def step_fn(carry, xs):
        probs, outcomes, actions = evolve_trials(agent, xs)

        deterministic('outcomes', outcomes)

        with plate('num_agents', Na):
            with plate('num_trials', Nt):
                sample('actions', dist.Categorical(logits=probs).to_event(1), obs=actions)
        
        return None, None
    
    # TODO: See if some information has to be passed from one block to the next and change init and carry accordingly
    init = None
    scan(step_fn, init, data, length=Nb)