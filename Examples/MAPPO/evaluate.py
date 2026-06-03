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
from vec_env import make_env as _make_single_env, PettingZooVecEnv, RedTeamVecEnv
from b_ace_py.red_team_policy import RedTeamPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_spec_path(path: str) -> str:
    """Convert a relative OS path to absolute so Godot can open it reliably."""
    if path.startswith("res://") or path.startswith("user://"):
        return path
    return str(Path(path).resolve())


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
            "combat_area": cfg.get("combat_area"),
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
                "fighter_spec":     _resolve_spec_path(a["blue"].get("fighter_spec", "res://assets/specs/default_fighter_spec.json")),
                "missile_spec":     _resolve_spec_path(a["blue"].get("missile_spec",  "res://assets/specs/default_missile_spec.json")),
            },
            "red_agents": {
                "num_agents":    a["red"]["num_agents"],
                "base_behavior": a["red"]["base_behavior"],
                "mission":       a["red"]["mission"],
                "init_position": a["red"]["init_position"],
                "init_hdg":      a["red"]["init_hdg"],
                "share_states":  a["red"].get("share_states", 1),
                "share_tracks":  a["red"].get("share_tracks", 1),
                "beh_config":    a["red"]["beh_config"],
                "fighter_spec":  _resolve_spec_path(a["red"].get("fighter_spec", "res://assets/specs/default_fighter_spec.json")),
                "missile_spec":  _resolve_spec_path(a["red"].get("missile_spec",  "res://assets/specs/default_missile_spec.json")),
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
            "eval.checkpoint is not set in eval_config.yaml.\n"
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
    b_ace_config = build_eval_b_ace_config(cfg)
    combat_area  = cfg.get("combat_area")
    red_cfg      = cfg["agents"]["red"]

    env_fns = [lambda: _make_single_env(b_ace_config, env_idx=0)]

    if red_cfg.get("base_behavior") == "external":
        # Mirror training: use RedTeamVecEnv so the actor only sees blue agents
        # (n_agents = n_blue) and the red team is driven by RedTeamPolicy.
        red_policy = RedTeamPolicy(
            obs_maps={},
            shot_threshold=red_cfg.get("shot_threshold", 0.85),
            shot_variation=red_cfg.get("shot_variation",  0.10),
            combat_area=combat_area,
        )
        venv = RedTeamVecEnv(env_fns, red_policy, combat_area=combat_area)
        print(f"  red policy  : RedTeamPolicy (external)  n_red={venv._n_red}")
    else:
        venv = PettingZooVecEnv(env_fns, combat_area=combat_area)
        print(f"  red policy  : {red_cfg['base_behavior']} (Godot FSM)")

    obs_dim    = venv.obs_dim
    action_dim = venv.action_dim
    n_agents   = venv.n_agents

    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  n_agents={n_agents}")
    if combat_area:
        print(f"  combat_area: x=[{combat_area['x_min']}, {combat_area['x_max']}]"
              f"  z=[{combat_area['z_min']}, {combat_area['z_max']}] NM  (+4 boundary obs)")
    print()

    # ── Load actor ────────────────────────────────────────────────────────
    hidden = cfg["algo"]["hidden_sizes"]
    actor  = MAPPOActor(obs_dim, action_dim, hidden)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    print(f"Loaded checkpoint (trained for {ckpt.get('global_step', '?'):,} steps)\n")

    # ── Episode loop ──────────────────────────────────────────────────────
    # VecEnv auto-resets on episode end. obs shape: (1, n_agents, obs_dim).
    results   = []
    obs       = venv.reset()   # (1, n_agents, obs_dim)
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
