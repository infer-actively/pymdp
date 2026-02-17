"""Generate legacy-style chained-cue navigation GIFs with the JAX environment.

This script recreates:
- `.github/chained_cue_navigation_v1.gif`
- `.github/chained_cue_navigation_v2.gif`

using `CueChainingEnv`, `Agent`, and environment assets in `pymdp/envs/assets/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from pymdp.agent import Agent
from pymdp.envs.cue_chaining import CueChainingEnv
from pymdp.envs.rollout import rollout


def build_agent(env: CueChainingEnv, policy_len: int = 4) -> Agent:
    """Construct an agent with known start location and uncertain latent context."""
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[3] = C[3].at[1].set(2.0)
    C[3] = C[3].at[2].set(-4.0)

    D_agent = [
        env.D[0],  # known initial location
        jnp.ones_like(env.D[1]) / env.D[1].shape[0],  # uncertain cue2 latent
        jnp.ones_like(env.D[2]) / env.D[2].shape[0],  # uncertain reward latent
    ]

    return Agent(
        A=env.A,
        B=env.B,
        C=C,
        D=D_agent,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        policy_len=policy_len,
        action_selection="deterministic",
        batch_size=1,
    )


def rollout_locations(
    cue2_state: int,
    reward_condition: int,
    num_timesteps: int = 12,
    seed: int = 0,
    policy_len: int = 4,
) -> tuple[list[tuple[int, int]], CueChainingEnv]:
    """Run one rollout and return location trajectory and environment."""
    env = CueChainingEnv(
        grid_shape=(5, 7),
        start_location=(0, 0),
        cue1_location=(2, 0),
        cue2_locations=((0, 2), (1, 3), (3, 3), (4, 2)),
        reward_locations=((1, 5), (3, 5)),
        cue2_state=cue2_state,
        reward_condition=reward_condition,
    )
    agent = build_agent(env, policy_len=policy_len)
    _, info = rollout(agent, env, num_timesteps=num_timesteps, rng_key=jr.PRNGKey(seed))
    locs = [env.index_to_coords(int(info["observation"][0][0, t, 0])) for t in range(num_timesteps + 1)]
    return locs, env


def interpolate_locations(
    locations: list[tuple[int, int]],
    points_per_segment: int = 10,
) -> np.ndarray:
    """Linearly interpolate between discrete trajectory points for smoother animation."""
    weights = np.linspace(0.0, 1.0, points_per_segment)
    segments = []
    for idx in range(len(locations) - 1):
        p0 = np.asarray(locations[idx], dtype=float)
        p1 = np.asarray(locations[idx + 1], dtype=float)
        inter = [(1.0 - w) * p0 + w * p1 for w in weights]
        segments.append(np.asarray(inter))
    return np.vstack(segments)


def load_assets(asset_dir: Path) -> dict[str, np.ndarray]:
    """Load and orient image assets."""
    mouse = plt.imread(asset_dir / "mouse.png")
    assets = {
        "mouse_down": mouse,
        "mouse_right": np.clip(ndimage.rotate(mouse, 90, reshape=True), 0.0, 1.0),
        "mouse_left": np.clip(ndimage.rotate(mouse, -90, reshape=True), 0.0, 1.0),
        "mouse_up": np.clip(ndimage.rotate(mouse, 180, reshape=True), 0.0, 1.0),
        "cheese": plt.imread(asset_dir / "cheese.png"),
        "shock": plt.imread(asset_dir / "shock.png"),
    }
    return assets


def make_animation(
    output_path: Path,
    env: CueChainingEnv,
    locations: list[tuple[int, int]],
    fps: int = 15,
    points_per_segment: int = 10,
) -> None:
    """Render one chained-cue navigation rollout to GIF."""
    cue1_loc = env.cue1_location
    cue2_loc = env.cue2_locations[env.D[1].argmax().item()]
    reward_condition = env.D[2].argmax().item()
    reward_locs = env.reward_locations
    reward_loc = reward_locs[reward_condition]
    shock_loc = reward_locs[1 - reward_condition]

    points = interpolate_locations(locations, points_per_segment=points_per_segment)

    rows, cols = env.grid_shape
    fig, ax = plt.subplots(figsize=(16, 10))
    X, Y = np.meshgrid(np.arange(cols + 1), np.arange(rows + 1))
    cue_grid = np.ones((rows, cols))
    cue_grid[cue1_loc[0], cue1_loc[1]] = 15.0
    for r, c in env.cue2_locations:
        cue_grid[r, c] = 5.0
    mesh = ax.pcolormesh(
        X,
        Y,
        cue_grid,
        edgecolors="k",
        linewidth=3,
        cmap="coolwarm",
        vmin=0,
        vmax=30,
    )
    ax.invert_yaxis()

    reward_top = ax.add_patch(
        patches.Rectangle(
            (reward_locs[0][1], reward_locs[0][0]),
            1.0,
            1.0,
            linewidth=10,
            edgecolor=[0.5, 0.5, 0.5],
            facecolor="none",
        )
    )
    reward_bottom = ax.add_patch(
        patches.Rectangle(
            (reward_locs[1][1], reward_locs[1][0]),
            1.0,
            1.0,
            linewidth=10,
            edgecolor=[0.5, 0.5, 0.5],
            facecolor="none",
        )
    )

    cue1_rect = ax.add_patch(
        patches.Rectangle(
            (cue1_loc[1], cue1_loc[0]),
            1.0,
            1.0,
            linewidth=15,
            edgecolor=[0.2, 0.7, 0.6],
            facecolor="none",
        )
    )
    cue2_rect = ax.add_patch(
        patches.Rectangle(
            (cue2_loc[1], cue2_loc[0]),
            1.0,
            1.0,
            linewidth=15,
            edgecolor=[0.2, 0.7, 0.6],
            facecolor="none",
        )
    )
    cue2_rect.set_visible(False)

    qmark_offsets = (0.4, 0.6)
    cue_texts = {}
    for i, (r, c) in enumerate(env.cue2_locations):
        cue_texts[(r, c)] = ax.text(c + qmark_offsets[0], r + qmark_offsets[1], "?", fontsize=55, color="k")
        if i == 0:
            pass

    ax.text(cue1_loc[1] + 0.3, cue1_loc[0] + 0.6, "Cue 1", fontsize=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    fig.tight_layout()

    assets = load_assets(Path(__file__).resolve().parents[2] / "pymdp" / "envs" / "assets")
    mouse_images = {
        "down": OffsetImage(assets["mouse_down"], zoom=0.05),
        "right": OffsetImage(assets["mouse_right"], zoom=0.05),
        "left": OffsetImage(assets["mouse_left"], zoom=0.05),
        "up": OffsetImage(assets["mouse_up"], zoom=0.05),
    }
    cheese_img = OffsetImage(assets["cheese"], zoom=0.03)
    shock_img = OffsetImage(assets["shock"], zoom=0.13)

    mouse_artists = {
        name: AnnotationBbox(img, (points[0, 1] + 0.5, points[0, 0] + 0.5), frameon=False)
        for name, img in mouse_images.items()
    }
    for name, artist in mouse_artists.items():
        ax.add_artist(artist)
        artist.set_visible(name == "down")

    revealed_cue1 = False
    revealed_cue2 = False
    cheese_artist = None
    shock_artist = None

    def _set_mouse_heading(prev_xy: np.ndarray, curr_xy: np.ndarray) -> None:
        dx = curr_xy[1] - prev_xy[1]
        dy = curr_xy[0] - prev_xy[0]
        if dx > 1e-6:
            heading = "right"
        elif dx < -1e-6:
            heading = "left"
        elif dy > 1e-6:
            heading = "down"
        elif dy < -1e-6:
            heading = "up"
        else:
            heading = next(name for name, artist in mouse_artists.items() if artist.get_visible())

        for name, artist in mouse_artists.items():
            artist.set_visible(name == heading)
            if name == heading:
                artist.xy = (curr_xy[1] + 0.5, curr_xy[0] + 0.5)
                artist.xybox = (curr_xy[1] + 0.5, curr_xy[0] + 0.5)
                artist.set_zorder(3)

    def init():
        return [mesh, *mouse_artists.values()]

    def update(frame_idx: int):
        nonlocal revealed_cue1, revealed_cue2, cheese_artist, shock_artist
        curr = points[frame_idx]
        prev = points[max(frame_idx - 1, 0)]
        curr_rc = (int(round(curr[0])), int(round(curr[1])))

        if (not revealed_cue1) and curr_rc == cue1_loc:
            revealed_cue1 = True
            cue1_rect.set_visible(False)
            cue2_rect.set_visible(True)
            cue_grid[cue2_loc[0], cue2_loc[1]] = 15.0
            mesh.set_array(cue_grid.ravel())
            for coord, text in cue_texts.items():
                if coord == cue2_loc:
                    text.set_position((cue2_loc[1] + 0.3, cue2_loc[0] + 0.6))
                    text.set_text("Cue 2")
                    text.set_fontsize(20)
                else:
                    text.set_visible(False)

        if (not revealed_cue2) and curr_rc == cue2_loc:
            revealed_cue2 = True
            cue2_rect.set_visible(False)
            if reward_condition == 0:
                reward_top.set_edgecolor("g")
                reward_top.set_facecolor("g")
                reward_bottom.set_edgecolor([0.7, 0.2, 0.2])
                reward_bottom.set_facecolor([0.7, 0.2, 0.2])
            else:
                reward_bottom.set_edgecolor("g")
                reward_bottom.set_facecolor("g")
                reward_top.set_edgecolor([0.7, 0.2, 0.2])
                reward_top.set_facecolor([0.7, 0.2, 0.2])

            cheese_artist = AnnotationBbox(
                cheese_img,
                (reward_loc[1] + 0.5, reward_loc[0] + 0.5),
                frameon=False,
            )
            cheese_artist.set_zorder(2)
            ax.add_artist(cheese_artist)

            shock_artist = AnnotationBbox(
                shock_img,
                (shock_loc[1] + 0.5, shock_loc[0] + 0.5),
                frameon=False,
            )
            shock_artist.set_zorder(2)
            ax.add_artist(shock_artist)

        _set_mouse_heading(prev, curr)
        artists = [mesh, *mouse_artists.values()]
        if cheese_artist is not None:
            artists.append(cheese_artist)
        if shock_artist is not None:
            artists.append(shock_artist)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(points),
        init_func=init,
        blit=True,
        interval=1000 / max(fps, 1),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate chained-cue navigation GIFs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".github"),
        help="Directory to write GIFs into.",
    )
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second.")
    parser.add_argument("--timesteps", type=int, default=12, help="Rollout timesteps.")
    parser.add_argument(
        "--interp-points",
        type=int,
        default=10,
        help="Interpolated points per step transition.",
    )
    args = parser.parse_args()

    # Legacy-matching variants:
    # v1 -> cue2 state 0 (Cue 1), reward condition 0 (TOP)
    # v2 -> cue2 state 2 (Cue 3), reward condition 1 (BOTTOM)
    variants = [
        ("chained_cue_navigation_v1.gif", 0, 0),
        ("chained_cue_navigation_v2.gif", 2, 1),
    ]

    for filename, cue2_state, reward_condition in variants:
        locations, env = rollout_locations(
            cue2_state=cue2_state,
            reward_condition=reward_condition,
            num_timesteps=args.timesteps,
            seed=0,
            policy_len=4,
        )
        make_animation(
            output_path=args.output_dir / filename,
            env=env,
            locations=locations,
            fps=args.fps,
            points_per_segment=args.interp_points,
        )
        print(f"Saved {(args.output_dir / filename).as_posix()}")


if __name__ == "__main__":
    main()
