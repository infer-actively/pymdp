from jax import numpy as jnp
from jax.lax import scan

class myClass(object):

    params: jnp.ndarray

    def __init__(self, params):
        self.params = params

    def update(self, delta_params):
        self.params += delta_params

myClass_instance = myClass(jnp.array([1., 2., 3.]))

def body(carry, t):
    myClass_instance.update(all_updates[t])
    return None, None

all_updates = jnp.ones((5, 3))

scan(body, None, jnp.arange(5))

# print out the values of the params after the scan
print(myClass_instance.params)