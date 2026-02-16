## Setup (inside the `pymdp` repo)

From the `pymdp` repository root, create a virtual environment

```bash
uv venv
uv sync
```

And then either edit the `pyproject.toml` to add dependency groups for gymnax, gymnasium or create a local, custom venv where you just
pip install those locally, e.g. 

```
uv pip install gymnasium, gymnax, pillow
````

maybe for gymnasium you should also install extra special groups like `gymnasium[box2d]` for bipedal walker.

## Run the eval script

```bash
python scripts/eval/cartpole_eval_harness.py \
  --seeds 0,1,2 \
  --num-episodes 200 \
  --inference-algo mmp \
  --inference-horizon 8 \
  --output-dir artifacts/cartpole_eval_mmp_h8 # might need to create an `artifacts` dir if not already there
```

## Run parity checks

```bash
python scripts/parity/check_env_parity.py
python scripts/parity/check_gymnax_block_rollout.py
python scripts/parity/check_single_step_parity.py
```

## Run dashboard GIF visualization

```bash
python scripts/eval/cartpole_dashboard_viz.py \
  --seed 0 \
  --num-episodes 200 \
  --inference-algo mmp \
  --inference-horizon 8 \
  --render-episodes last \
  --output-dir artifacts/cartpole_dashboard_viz
```
