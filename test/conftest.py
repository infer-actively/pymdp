import os

def pytest_configure(config):
    """
    Configure JAX memory settings for parallel test execution.

    This prevents OOM errors when running tests in parallel with pytest-xdist
    by limiting JAX's memory preallocation behavior.

    Additionally we enable JAX compilation caching for improved repeated test performance.

    See: 
    - https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    - https://docs.jax.dev/en/latest/persistent_compilation_cache.html
    """
    # Use platform memory allocator (more conservative than default behaviour)
    if 'XLA_PYTHON_CLIENT_ALLOCATOR' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # Limit memory fraction per device (to prevent OOM kills)
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.20'

    # Enable JAX compilation cache for faster repeated runs
    if 'JAX_COMPILATION_CACHE_DIR' not in os.environ:
        os.environ['JAX_COMPILATION_CACHE_DIR'] = '.jax_cache'
