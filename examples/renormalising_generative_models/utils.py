from jax import numpy as jnp

from sklearn.datasets import fetch_openml
from typing import Tuple

def load_mnist(cache_dir: str= './mnist_cache') -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=cache_dir, parser='auto')

    x, y = mnist.data, mnist.target

    x = x.values if hasattr(x, 'values') else x
    y = y.values if hasattr(y, 'values') else y

    x = x.astype(jnp.float32)
    y = y.astype(jnp.int32)

    x = x.reshape(-1, 28, 28)

    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_test), jnp.array(y_test)

def extract_exemplars(images, labels, m_per_class, n_classes):
    """Extract the first m_per_class images for each digit class."""
    exemplar_indices = []
    for digit in range(n_classes):
        indices = jnp.where(labels == digit)[0][:m_per_class]
        exemplar_indices.append(indices)
    exemplar_indices = jnp.concatenate(exemplar_indices)
    return images[exemplar_indices], labels[exemplar_indices], exemplar_indices