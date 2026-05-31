"""
PettingZooVecEnv: runs N B-ACE Godot processes in parallel using threads.

Each Godot instance gets its own TCP port (base_port + env_idx) so there
are no port collisions even when running on the same machine.

Why threads (not processes):
    B-ACE's bottleneck is TCP socket I/O — waiting for Godot to respond.
    Python releases the GIL during socket I/O, so threads genuinely run
    in parallel. Each env instance owns its own socket, so there is no
    shared state between threads.

Output shapes (n_envs=E, n_agents=A, obs_dim=O, action_dim=D):
    reset() -> obs:     (E, A, O)
    step()  -> obs:     (E, A, O)
                rewards: (E,)
                dones:   (E,)
                infos:   list[dict]  length E
"""
import copy
import random
from concurrent.futures import ThreadPoolExecutor
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
    Vectorised B-ACE environment that steps all Godot processes in parallel.
    """

    def __init__(self, env_fns: list[Callable[[], B_ACE_GodotPettingZooWrapper]]):
        self.envs   = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        ref = self.envs[0]
        self.agents         = ref.possible_agents
        self.n_agents       = len(self.agents)
        self.obs_dim        = ref.observation_space.shape[0]
        self.action_dim     = ref.action_space.shape[0]
        self.global_obs_dim = self.obs_dim * self.n_agents

        # One worker per env so all can step simultaneously
        self._pool = ThreadPoolExecutor(max_workers=self.n_envs)

    # ------------------------------------------------------------------
    def _extract(self, obs_dict: dict) -> np.ndarray:
        """Convert {agent: {"obs": ndarray, ...}} -> (n_agents, obs_dim)."""
        return np.stack([obs_dict[a]["obs"] for a in self.agents], axis=0)

    def _reset_one(self, env: B_ACE_GodotPettingZooWrapper) -> np.ndarray:
        obs_dict, _ = env.reset()
        return self._extract(obs_dict)

    def _step_one(
        self, env: B_ACE_GodotPettingZooWrapper, actions_dict: dict
    ) -> tuple[np.ndarray, float, float, dict]:
        obs_dict, reward, terminated, truncated, info = env.step(actions_dict)
        done = terminated or truncated
        if done:
            obs_dict, _ = env.reset()
        obs = self._extract(obs_dict)
        info_out = (
            {"terminated": terminated, "truncated": truncated, **info}
            if isinstance(info, dict)
            else {"terminated": terminated, "truncated": truncated}
        )
        return obs, float(reward), float(done), info_out

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset all envs in parallel. Returns obs shape (E, A, O)."""
        futures  = [self._pool.submit(self._reset_one, env) for env in self.envs]
        obs_list = [f.result() for f in futures]
        return np.stack(obs_list, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step all envs in parallel.

        actions: (n_envs, n_agents, action_dim)
        Returns: obs (E,A,O), rewards (E,), dones (E,), infos list[dict]
        """
        futures = []
        for i, env in enumerate(self.envs):
            actions_dict = {
                a: actions[i, j].tolist()
                for j, a in enumerate(self.agents)
            }
            futures.append(self._pool.submit(self._step_one, env, actions_dict))

        results     = [f.result() for f in futures]
        obs_list, reward_list, done_list, info_list = zip(*results)

        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list,   dtype=np.float32),
            list(info_list),
        )

    def get_global_obs(self, obs: np.ndarray) -> np.ndarray:
        """obs (E, A, O) -> global_obs (E, A*O)."""
        return obs.reshape(obs.shape[0], -1)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for env in self.envs:
            env.close()
