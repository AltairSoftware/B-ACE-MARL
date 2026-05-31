"""
PettingZooVecEnv: runs N B-ACE Godot processes in sequence and presents
them as a single vectorised environment to the training loop.

Each Godot instance gets its own TCP port (base_port + env_idx) so there
are no collisions, even when run on the same machine.
"""
import copy
import random
from typing import Callable

import numpy as np

from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper


def make_env(b_ace_config: dict, env_idx: int) -> B_ACE_GodotPettingZooWrapper:
    """Factory: creates one environment with a unique port and random seed."""
    cfg = copy.deepcopy(b_ace_config)
    cfg["EnvConfig"]["port"] = cfg["EnvConfig"].get("port", 12500) + env_idx
    cfg["EnvConfig"]["seed"] = random.randint(0, 1_000_000)
    return B_ACE_GodotPettingZooWrapper(device="cpu", **cfg)


class PettingZooVecEnv:
    """
    Thin wrapper that runs `n_envs` B-ACE environments sequentially
    (DummyVectorEnv style) and stacks their outputs into batch arrays.

    Output shapes (n_envs=E, n_agents=A, obs_dim=O, action_dim=D):
        reset()  -> obs:     (E, A, O)
        step()   -> obs:     (E, A, O)
                    rewards: (E,)
                    dones:   (E,)
                    infos:   list[dict]  length E
    """

    def __init__(self, env_fns: list[Callable[[], B_ACE_GodotPettingZooWrapper]]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        ref = self.envs[0]
        self.agents    = ref.possible_agents
        self.n_agents  = len(self.agents)
        self.obs_dim   = ref.observation_space.shape[0]
        self.action_dim = ref.action_space.shape[0]
        self.global_obs_dim = self.obs_dim * self.n_agents

    # ------------------------------------------------------------------
    def _extract(self, obs_dict: dict) -> np.ndarray:
        """Convert {agent: {"obs": ndarray, ...}} -> (n_agents, obs_dim)."""
        return np.stack([obs_dict[a]["obs"] for a in self.agents], axis=0)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset all envs. Returns obs shape (n_envs, n_agents, obs_dim)."""
        obs_list = []
        for env in self.envs:
            obs_dict, _ = env.reset()
            obs_list.append(self._extract(obs_dict))
        return np.stack(obs_list, axis=0)   # (E, A, O)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        actions: (n_envs, n_agents, action_dim)
        Returns: obs (E,A,O), rewards (E,), dones (E,), infos list[dict]
        """
        obs_list     = []
        reward_list  = []
        done_list    = []
        info_list    = []

        for i, env in enumerate(self.envs):
            actions_dict = {
                a: actions[i, j].tolist()
                for j, a in enumerate(self.agents)
            }
            obs_dict, reward, terminated, truncated, info = env.step(actions_dict)
            done = terminated or truncated

            if done:
                obs_dict, _ = env.reset()

            obs_list.append(self._extract(obs_dict))
            reward_list.append(float(reward))
            done_list.append(float(done))
            info_list.append({"terminated": terminated, "truncated": truncated,
                               **info} if isinstance(info, dict) else
                              {"terminated": terminated, "truncated": truncated})

        return (
            np.stack(obs_list,    axis=0),   # (E, A, O)
            np.array(reward_list, dtype=np.float32),  # (E,)
            np.array(done_list,   dtype=np.float32),  # (E,)
            info_list,
        )

    def get_global_obs(self, obs: np.ndarray) -> np.ndarray:
        """obs (E, A, O) -> global_obs (E, A*O)."""
        E = obs.shape[0]
        return obs.reshape(E, -1)

    def close(self) -> None:
        for env in self.envs:
            env.close()
