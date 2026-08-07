"""Evaluation strategies for the MNIST RGM.

Two named strategies make explicit that our benchmark accuracy and SPM's
reported 95.1% measure *different things* (see PR #414 review). Quoting one
number without the strategy invites a false "we're behind" comparison.

- ``"spm_parity"`` — reproduces SPM's ``DEM_MNIST_RGM`` protocol so the port can
  be validated against the reference. The SVD bases and bin ranges are fit on
  all 11,000 images (the 10k active-learning set plus the 1k evaluation slice)
  *before* the split, and evaluation uses the in-distribution training slice
  ``training[N_TRAIN : N_TRAIN + N_EVAL]``. This is a correctness check, not a
  generalisation benchmark: the preprocessing sees the eval images and the eval
  set is drawn from MNIST's train split.

- ``"benchmark"`` — the honest generalisation measure. The front-end is fit on
  training data only (via the structure exemplars; ``x_basis=None`` so
  ``RGMHierarchy.from_exemplars`` never sees test data), and evaluation uses the
  *complete* official MNIST test split.

Both strategies share the same active-learning set and the same 13-per-class
structure exemplars; they differ only in what fits the front-end and what is
evaluated.
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

from preprocess import preprocess

SPM_PARITY = "spm_parity"
BENCHMARK = "benchmark"
STRATEGIES = (SPM_PARITY, BENCHMARK)


class StrategyData(NamedTuple):
    """Preprocessed arrays and labels for one evaluation strategy."""

    name: str
    description: str
    x_basis: jnp.ndarray | None  # SVD-basis fit set (None -> fit on exemplars)
    x_train: jnp.ndarray         # active-learning images
    y_train: np.ndarray
    x_eval: jnp.ndarray          # evaluation images
    y_eval: np.ndarray
    eval_label: str              # human-readable description of the eval set


def make_strategy(
    strategy: str,
    x_train_raw: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test_raw: jnp.ndarray,
    y_test: jnp.ndarray,
    n_train: int = 10_000,
    n_eval: int = 1_000,
    preprocess_fn: Callable = preprocess,
) -> StrategyData:
    """Build the (basis, train, eval) split for one evaluation strategy.

    Args:
        strategy: ``"spm_parity"`` or ``"benchmark"``.
        x_train_raw, y_train: raw MNIST training images/labels.
        x_test_raw, y_test: raw MNIST official test images/labels.
        n_train: number of active-learning images (SPM uses 10,000).
        n_eval: size of the in-distribution eval slice for parity mode
            (SPM uses 1,000). Ignored by benchmark mode, which uses all of test.
        preprocess_fn: image preprocessing (fit-free; per-image resize/pad).

    Returns:
        StrategyData with preprocessed arrays.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_train_used = preprocess_fn(x_train_raw[:n_train])
    y_train_used = y_train[:n_train]

    if strategy == SPM_PARITY:
        # Fit the front-end on all 11k (train set + eval slice) before splitting,
        # then evaluate on the in-distribution training slice.
        eval_slice = slice(n_train, n_train + n_eval)
        x_basis = preprocess_fn(x_train_raw[: n_train + n_eval])
        x_eval = preprocess_fn(x_train_raw[eval_slice])
        y_eval = y_train[eval_slice]
        return StrategyData(
            name=SPM_PARITY,
            description=(
                "SPM DEM_MNIST_RGM parity: SVD bases/bins fit on all "
                f"{n_train + n_eval} images before the split; eval on the "
                "in-distribution training slice."
            ),
            x_basis=x_basis,
            x_train=x_train_used,
            y_train=y_train_used,
            x_eval=x_eval,
            y_eval=y_eval,
            eval_label=f"training[{n_train}:{n_train + n_eval}] (in-distribution)",
        )

    # benchmark: train-only front-end, full official test set
    x_eval = preprocess_fn(x_test_raw)
    y_eval = y_test
    return StrategyData(
        name=BENCHMARK,
        description=(
            "Clean benchmark: front-end fit on training data only (structure "
            "exemplars); eval on the full official MNIST test set."
        ),
        x_basis=None,
        x_train=x_train_used,
        y_train=y_train_used,
        x_eval=x_eval,
        y_eval=y_eval,
        eval_label=f"official MNIST test set ({len(y_test)} held-out)",
    )
