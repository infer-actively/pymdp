import numpyro.distributions as dist
from jax import lax
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan
from typing import Any

def evolve_trials(agent: Any, data: Any) -> Any:

    def step_fn(carry: Any, xs: dict[str, Any]) -> tuple[Any, tuple[Any, Any, Any]]:
        empirical_prior = carry
        outcomes = xs['outcomes']
        qs = agent.infer_states(outcomes, empirical_prior)
        q_pi, _ = agent.infer_policies(qs)

        probs = agent.action_probabilities(q_pi)

        actions = xs['actions']
        empirical_prior = agent.update_empirical_prior(actions, qs)
        #TODO: if outcomes and actions are None, generate samples
        return empirical_prior, (probs, outcomes, actions)

    prior = agent.D
    _, res = lax.scan(step_fn, prior, data)

    return res

def aif_likelihood(Nb: int, Nt: int, Na: int, data: Any, agent: Any) -> None:
    # Na -> batch dimension - number of different subjects/agents
    # Nb -> number of experimental blocks
    # Nt -> number of trials within each block

    def step_fn(carry: Any, xs: Any) -> tuple[None, None]:
        probs, outcomes, actions = evolve_trials(agent, xs)

        deterministic('outcomes', outcomes)

        with plate('num_agents', Na):
            with plate('num_trials', Nt):
                sample('actions', dist.Categorical(logits=probs).to_event(1), obs=actions)
        
        return None, None
    
    # TODO: See if some information has to be passed from one block to the next and change init and carry accordingly
    init = None
    scan(step_fn, init, data, length=Nb)
