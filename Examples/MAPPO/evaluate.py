"""
B-ACE MAPPO — evaluation script.

Loads a trained actor from a checkpoint and runs the policy in the
B-ACE environment for visual inspection.

Before running:
  1. Set eval.checkpoint in eval_config.yaml to the path of the saved model.
  2. Adjust eval.n_episodes and eval.deterministic as needed.
  3. env.renderize and env.speed_up are already set for visual inspection.

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
from vec_env import make_env as _make_single_env, PettingZooVecEnv


# ---------------------------------------------------------------------------
# Evaluation config builder
#
# Sends only env and agent parameters to Godot.
# RewardsConfig is intentionally omitted so Godot uses its own defaults
# from Default_Sim_Config.json without any override.
# ---------------------------------------------------------------------------

def build_eval_b_ace_config(cfg: dict) -> dict:
    e = cfg["env"]
    a = cfg["agents"]
    return {
        "EnvConfig": {
            "env_path":    e["env_path"],
            "renderize":   e["renderize"],
            "speed_up":    e["speed_up"],
            "max_cycles":  e["max_cycles"],
            "seed":        e["seed"],
            "action_type": e["action_type"],
        },
        "AgentsConfig": {
            "blue_agents": {
                "num_agents":       a["blue"]["num_agents"],
                "base_behavior":    a["blue"]["base_behavior"],
                "mission":          a["blue"]["mission"],
                "init_position":    a["blue"]["init_position"],
                "init_hdg":         a["blue"]["init_hdg"],
                "target_position":  a["blue"]["target_position"],
                "rnd_offset_range": a["blue"]["rnd_offset_range"],
                "fighter_spec":     a["blue"].get("fighter_spec", "res://assets/specs/default_fighter_spec.json"),
                "missile_spec":     a["blue"].get("missile_spec",  "res://assets/specs/default_missile_spec.json"),
            },
            "red_agents": {
                "num_agents":    a["red"]["num_agents"],
                "base_behavior": a["red"]["base_behavior"],
                "mission":       a["red"]["mission"],
                "init_position": a["red"]["init_position"],
                "init_hdg":      a["red"]["init_hdg"],
                "beh_config":    a["red"]["beh_config"],
                "fighter_spec":  a["red"].get("fighter_spec", "res://assets/specs/default_fighter_spec.json"),
                "missile_spec":  a["red"].get("missile_spec",  "res://assets/specs/default_missile_spec.json"),
            },
        },
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate() -> None:
    with open("eval_config.yaml", encoding="utf-8") as f:
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

    # ── Environment ───────────────────────────────────────────────────────
    # Use PettingZooVecEnv (n_envs=1) so that combat_area observation
    # extension is applied consistently with training.
    b_ace_config = build_eval_b_ace_config(cfg)
    combat_area  = cfg.get("combat_area")
    venv = PettingZooVecEnv(
        [lambda: _make_single_env(b_ace_config, env_idx=0)],
        combat_area=combat_area,
    )

    obs_dim    = venv.obs_dim
    action_dim = venv.action_dim
    n_agents   = venv.n_agents

    print(f"obs_dim={obs_dim}  action_dim={action_dim}  n_agents={n_agents}")
    if combat_area:
        print(f"combat_area obs extension: +4 boundary values applied")

    # ── Load actor ────────────────────────────────────────────────────────
    hidden = cfg["algo"]["hidden_sizes"]
    actor  = MAPPOActor(obs_dim, action_dim, hidden)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    print(f"Loaded checkpoint (trained for {ckpt.get('global_step', '?'):,} steps)\n")

    # ── Episode loop ──────────────────────────────────────────────────────
    # VecEnv auto-resets on episode end. obs shape: (1, n_agents, obs_dim).
    results  = []
    obs = venv.reset()   # (1, n_agents, obs_dim)

    ep_reward = 0.0
    ep_len    = 0
    ep        = 0

    while ep < n_episodes:
        obs_agents = obs[0]   # (n_agents, obs_dim)
        obs_t = torch.as_tensor(obs_agents, dtype=torch.float32)

        with torch.no_grad():
            if deterministic:
                actions_t = actor.get_deterministic_action(obs_t)
            else:
                actions_t, _, _ = actor.get_action(obs_t)

        # VecEnv expects (n_envs, n_agents, action_dim)
        actions = actions_t.numpy()[np.newaxis, :]   # (1, n_agents, action_dim)
        obs, rewards, dones, _ = venv.step(actions)

        ep_reward += float(rewards[0])
        ep_len    += 1

        if dones[0]:
            ep += 1
            outcome = "WIN " if ep_reward > 0 else "LOSE"
            print(
                f"  ep={ep:3d}  {outcome}  "
                f"len={ep_len:4d}  reward={ep_reward:8.3f}"
            )
            results.append({"reward": ep_reward, "length": ep_len})
            # VecEnv has already auto-reset; obs is already the new episode's first obs
            ep_reward = 0.0
            ep_len    = 0

    # ── Summary ───────────────────────────────────────────────────────────
    rewards_list = [r["reward"] for r in results]
    lengths      = [r["length"] for r in results]
    wins         = sum(1 for r in rewards_list if r > 0)

    print()
    print("-" * 40)
    print(f"Episodes  : {n_episodes}")
    print(f"Win rate  : {wins}/{n_episodes} ({100 * wins / n_episodes:.0f}%)")
    print(f"Avg reward: {sum(rewards_list) / len(rewards_list):.3f}")
    print(f"Avg length: {sum(lengths) / len(lengths):.0f} steps")
    print("-" * 40)

    venv.close()


if __name__ == "__main__":
    evaluate()
