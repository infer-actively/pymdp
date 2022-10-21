import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan

def evolve_trials(agent, data):

    def step_fn(carry, xs):
        outcome = xs['outcomes']
        qx = agent.infer_states(outcome)
        q_pi, _ = agent.infer_policies()
        
        nc = agent.num_controls
        num_factors = len(agent.num_controls)
        
        marginal = []
        for factor_i in range(num_factors):
            m = []
            actions = agent.policies[:, 0, factor_i]
            for a in range(nc[factor_i]):
                m.append( jnp.where(actions==a, q_pi, 0).sum() )
            marginal.append(jnp.stack(m))

        action = xs['actions']
        agent.update_empirical_prior(action)
        #TODO: if outcomes and actions are None, generate samples
        return None, (marginal, outcome, action)

    _, res = lax.scan(step_fn, None, data)

    return res[0], res[1], res[2]

def aif_likelihood(Na, Nb, Nt, data, agent):
    # Na -> batch dimension - number of different subjects/agents
    # Nb -> number of experimental blocks
    # Nt -> number of trials within each block

    def step_fn(carry, xs):
        probs, outcomes, actions = evolve_trials(agent, xs)

        probs = 0.5*jnp.ones((2, 2))
        print(probs.shape)

        # deterministic('outcomes', outcomes)

        with plate('num_agents', Na):
            with plate('num_trials', Nt):
                sample('actions', dist.Categorical(probs=probs).to_event(1))
        
        return None, None
    
    # TODO: See if some information has to be passed from one block to the next and change init and carry accordingly
    init = None
    scan(step_fn, init, data, length=Nb)