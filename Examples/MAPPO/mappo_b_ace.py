"""
B-ACE MAPPO — user entry point.

Customize this file to change:
  * Scenario / agent configuration  ->  build_b_ace_config()  or  config.yaml
  * Reward shaping                  ->  reward_fn()
  * Hyperparameters                 ->  config.yaml

All PPO algorithm internals are in ppo.py (no need to edit).

Usage:
    python mappo_b_ace.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup — keeps relative paths in config.yaml valid regardless of where
# the script is launched from
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent.resolve()
os.chdir(_script_dir)

_project_root = _script_dir.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ppo import train  # noqa: E402


# ---------------------------------------------------------------------------
# Environment configuration bridge
#
# Maps the flat YAML structure to the nested dict expected by
# B_ACE_GodotPettingZooWrapper.
#
# Edit this function when you add new environment options to config.yaml.
# ---------------------------------------------------------------------------

def build_b_ace_config(cfg: dict) -> dict:
    e = cfg["env"]
    a = cfg["agents"]
    r = cfg["rewards"]
    return {
        "EnvConfig": {
            "env_path":    e["env_path"],
            "renderize":   e["renderize"],
            "speed_up":    e["speed_up"],
            "max_cycles":  e["max_cycles"],
            "seed":        e["seed"],
            "action_type": e["action_type"],
            "RewardsConfig": {
                "mission_factor":              r["mission_factor"],
                "missile_fire_factor":         r["missile_fire_factor"],
                "missile_no_fire_factor":      r["missile_no_fire_factor"],
                "missile_miss_factor":         r["missile_miss_factor"],
                "detect_loss_factor":          r["detect_loss_factor"],
                "keep_track_factor":           r["keep_track_factor"],
                "hit_enemy_factor":            r["hit_enemy_factor"],
                "hit_own_factor":              r["hit_own_factor"],
                "mission_accomplished_factor": r["mission_accomplished_factor"],
            },
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
            },
            "red_agents": {
                "num_agents":    a["red"]["num_agents"],
                "base_behavior": a["red"]["base_behavior"],
                "mission":       a["red"]["mission"],
                "init_position": a["red"]["init_position"],
                "init_hdg":      a["red"]["init_hdg"],
                "beh_config":    a["red"]["beh_config"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Reward shaping hook
#
# Called after every environment step for each parallel environment.
# Edit this function to add Python-side reward shaping on top of the
# rewards already configured in config.yaml.
#
# Args:
#   obs    Next observation, shape (n_agents, obs_dim).
#   reward Raw reward from the Godot simulation (sum over all blue agents).
#   done   True if the episode ended on this step.
#
# Returns:
#   Modified reward (float).
#
# Example — small penalty per step to encourage shorter episodes:
#   def reward_fn(obs, reward, done):
#       return reward - 0.001
# ---------------------------------------------------------------------------

def reward_fn(obs: np.ndarray, reward: float, done: bool) -> float:
    _ = obs, done   # unused in default implementation; use them when adding shaping
    return reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    restore = cfg["logging"].get("restore")

    train(
        cfg,
        build_b_ace_config(cfg),
        reward_fn=reward_fn,
        restore_path=restore,
    )
