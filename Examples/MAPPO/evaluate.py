"""
B-ACE MAPPO — evaluation script.

Loads a trained actor from a checkpoint and runs the policy in the
B-ACE environment for visual inspection.

Before running:
  1. Set eval.checkpoint in config.yaml to the path of the saved model.
  2. Set env.renderize to 1 in config.yaml to see the Godot window.
  3. Adjust eval.n_episodes and eval.deterministic as needed.

Usage:
    python evaluate.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent.resolve()
os.chdir(_script_dir)

_project_root = _script_dir.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from networks import MAPPOActor
from vec_env import make_env as _make_single_env
from mappo_b_ace import build_b_ace_config


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    n_episodes      = eval_cfg.get("n_episodes", 10)
    deterministic   = eval_cfg.get("deterministic", True)

    if not checkpoint_path:
        raise ValueError(
            "eval.checkpoint is not set in config.yaml.\n"
            "Set it to the path of a saved model, e.g.:\n"
            "  eval:\n"
            "    checkpoint: Results/mappo_b_ace_xxx/final.pt"
        )

    print("=" * 60)
    print("B-ACE MAPPO Evaluation")
    print("=" * 60)
    print(f"  checkpoint  : {checkpoint_path}")
    print(f"  n_episodes  : {n_episodes}")
    print(f"  deterministic: {deterministic}")
    print(f"  renderize   : {cfg['env']['renderize']}")
    print()

    # ── Environment (single, no VecEnv needed) ────────────────────────────
    b_ace_config = build_b_ace_config(cfg)
    env = _make_single_env(b_ace_config, env_idx=0)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    n_agents   = len(env.possible_agents)

    print(f"obs_dim={obs_dim}  action_dim={action_dim}  n_agents={n_agents}")

    # ── Load actor ────────────────────────────────────────────────────────
    hidden = cfg["algo"]["hidden_sizes"]
    actor  = MAPPOActor(obs_dim, action_dim, hidden)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    print(f"Loaded checkpoint (trained for {ckpt.get('global_step', '?'):,} steps)\n")

    # ── Episode loop ──────────────────────────────────────────────────────
    results = []

    for ep in range(1, n_episodes + 1):
        obs_dict, _ = env.reset()
        obs = np.stack([obs_dict[a]["obs"] for a in env.possible_agents], axis=0)

        ep_reward = 0.0
        ep_len    = 0

        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                if deterministic:
                    actions_t = actor.get_deterministic_action(obs_t)
                else:
                    actions_t, _, _ = actor.get_action(obs_t)

            actions_np   = actions_t.numpy()
            actions_dict = {
                a: actions_np[i].tolist()
                for i, a in enumerate(env.possible_agents)
            }

            obs_dict, reward, terminated, truncated, _ = env.step(actions_dict)
            obs = np.stack([obs_dict[a]["obs"] for a in env.possible_agents], axis=0)

            ep_reward += float(reward)
            ep_len    += 1

            if terminated or truncated:
                outcome = "WIN " if reward > 0 else "LOSE"
                print(
                    f"  ep={ep:3d}  {outcome}  "
                    f"len={ep_len:4d}  reward={ep_reward:8.3f}"
                )
                results.append({"reward": ep_reward, "length": ep_len})
                break

    # ── Summary ───────────────────────────────────────────────────────────
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    wins    = sum(1 for r in rewards if r > 0)

    print()
    print("-" * 40)
    print(f"Episodes  : {n_episodes}")
    print(f"Win rate  : {wins}/{n_episodes} ({100 * wins / n_episodes:.0f}%)")
    print(f"Avg reward: {sum(rewards) / len(rewards):.3f}")
    print(f"Avg length: {sum(lengths) / len(lengths):.0f} steps")
    print("-" * 40)

    env.close()


if __name__ == "__main__":
    evaluate()
