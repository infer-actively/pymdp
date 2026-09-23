"""Run both evaluation strategies and report their (non-comparable) numbers.

Builds and trains a separate hierarchy per strategy — spm_parity fits the SVD
front-end on all 11k images, benchmark fits it on the structure exemplars only —
then evaluates each on its own eval set. Prints both clearly-labeled numbers so
the parity check and the honest benchmark are never conflated.

Usage:
    uv run python compare_strategies.py [N_TRAIN] [N_EVAL_SLICE]
"""

import sys
import time

import numpy as np

from utils import load_mnist, extract_exemplars
from preprocess import preprocess
from discretise import DiscretiseConfig
from rgm_hierarchy import RGMHierarchy
from strategies import make_strategy, STRATEGIES

M_PER_CLASS = 13
NUM_CLASSES = 10


def run_strategy(strat, x_exemplars, y_exemplars, config):
    rgm = RGMHierarchy.from_exemplars(
        x_exemplars, y_exemplars, config, x_basis=strat.x_basis
    )
    t0 = time.time()
    rgm.train(
        strat.x_train, strat.y_train,
        concentration_lower=1 / 16, concentration_cls=1 / 128,
        lr_pA=1.0, beta=512.0, eta=512.0, num_iter=1,
        log_every=2000, scan_chunk_size=50,
    )
    dt = time.time() - t0
    pred, _, _ = rgm.classify(strat.x_eval)
    acc = float((np.asarray(pred) == np.asarray(strat.y_eval)).mean())
    print(f"[{strat.name}] acc={acc:.2%} on {strat.eval_label} ({dt:.0f}s)", flush=True)
    return acc


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000

    print(f"[setup] N_TRAIN={n_train} N_EVAL_SLICE={n_eval}", flush=True)
    x_train, y_train, x_test, y_test = load_mnist()

    x_exemplars_raw, y_exemplars, _ = extract_exemplars(
        x_train, y_train, M_PER_CLASS, NUM_CLASSES
    )
    x_exemplars = preprocess(x_exemplars_raw)
    config = DiscretiseConfig()

    results = {}
    for strategy in STRATEGIES:
        strat = make_strategy(strategy, x_train, y_train, x_test, y_test,
                              n_train=n_train, n_eval=n_eval)
        print(f"\n===== {strategy} =====\n{strat.description}", flush=True)
        results[strategy] = run_strategy(strat, x_exemplars, y_exemplars, config)

    print("\n================ SUMMARY ================", flush=True)
    for strategy, acc in results.items():
        print(f"{strategy:<11}: {acc:.2%}", flush=True)
    print("\nThese numbers are NOT comparable: spm_parity fits preprocessing on", flush=True)
    print("its eval slice and evaluates in-distribution; benchmark is held-out.", flush=True)


if __name__ == "__main__":
    main()
