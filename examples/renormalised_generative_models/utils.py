from jax import numpy as jnp


def extract_exemplars(images, labels, m_per_class, n_classes):
    """Extract the first m_per_class images for each digit class."""
    exemplar_indices = []
    for digit in range(n_classes):
        indices = jnp.where(labels == digit)[0][:m_per_class]
        exemplar_indices.append(indices)
    exemplar_indices = jnp.concatenate(exemplar_indices)
    return images[exemplar_indices], labels[exemplar_indices], exemplar_indices