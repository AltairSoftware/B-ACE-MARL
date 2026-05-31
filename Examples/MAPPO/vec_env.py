"""
PettingZooVecEnv: runs N B-ACE Godot processes in parallel using threads.

Each Godot instance gets its own TCP port (base_port + env_idx) so there
are no port collisions even when running on the same machine.

Why threads (not processes):
    B-ACE's bottleneck is TCP socket I/O — waiting for Godot to respond.
    Python releases the GIL during socket I/O, so threads genuinely run
    in parallel. Each env instance owns its own socket, so there is no
    shared state between threads.

Combat area observation extension (optional):
    When combat_area is provided, 4 normalized boundary distance values are
    appended to each agent's observation vector:
        [dist_to_x_min, dist_to_x_max, dist_to_z_min, dist_to_z_max]
    Positive = inside the boundary, negative = outside.
    obs[0] = own_x_pos, obs[1] = own_z_pos (confirmed by probe).

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

    Args:
        env_fns:      List of callables that each return a B-ACE env instance.
        combat_area:  Optional dict with keys x_min, x_max, z_min, z_max.
                      When provided, 4 boundary distance values are appended
                      to each agent's observation (obs_dim increases by 4).
    """

    # Indices of position components in the raw Godot observation
    # (confirmed by probe: own_x_pos=0, own_z_pos=1)
    _X_IDX = 0
    _Z_IDX = 1

    def __init__(
        self,
        env_fns: list[Callable[[], B_ACE_GodotPettingZooWrapper]],
        combat_area: dict | None = None,
    ):
        self.envs   = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        ref = self.envs[0]
        self.agents      = ref.possible_agents
        self.n_agents    = len(self.agents)
        self.action_dim  = ref.action_space.shape[0]

        base_obs_dim     = ref.observation_space.shape[0]
        self._combat_area = combat_area

        # 4 boundary distance values appended per agent when area is active
        self.obs_dim        = base_obs_dim + (4 if combat_area else 0)
        self.global_obs_dim = self.obs_dim * self.n_agents

        # One worker per env so all can step simultaneously
        self._pool = ThreadPoolExecutor(max_workers=self.n_envs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, obs_dict: dict) -> np.ndarray:
        """Convert {agent: {"obs": ndarray, ...}} -> (n_agents, base_obs_dim)."""
        return np.stack([obs_dict[a]["obs"] for a in self.agents], axis=0)

    def _append_boundary_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Append 4 boundary distance values to each agent's observation.

        obs: (n_agents, base_obs_dim)
        Returns: (n_agents, base_obs_dim + 4)

        Appended values (positive = inside boundary, negative = outside):
            [dist_to_x_min, dist_to_x_max, dist_to_z_min, dist_to_z_max]
        All normalized by the area's width/depth so the scale is ~[-1, 1].
        """
        ca = self._combat_area
        x = obs[:, self._X_IDX]          # own_x_pos, shape (n_agents,)
        z = obs[:, self._Z_IDX]          # own_z_pos, shape (n_agents,)

        area_w = ca["x_max"] - ca["x_min"]
        area_d = ca["z_max"] - ca["z_min"]

        boundary = np.column_stack([
            (x - ca["x_min"]) / area_w,   # dist to x_min (left)
            (ca["x_max"] - x) / area_w,   # dist to x_max (right)
            (z - ca["z_min"]) / area_d,   # dist to z_min (near)
            (ca["z_max"] - z) / area_d,   # dist to z_max (far)
        ])                                # shape (n_agents, 4)

        return np.concatenate([obs, boundary], axis=1)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply combat area extension if configured."""
        if self._combat_area is not None:
            obs = self._append_boundary_obs(obs)
        return obs

    def _reset_one(self, env: B_ACE_GodotPettingZooWrapper) -> np.ndarray:
        obs_dict, _ = env.reset()
        return self._process_obs(self._extract(obs_dict))

    def _step_one(
        self, env: B_ACE_GodotPettingZooWrapper, actions_dict: dict
    ) -> tuple[np.ndarray, float, float, dict]:
        obs_dict, reward, terminated, truncated, info = env.step(actions_dict)
        done = terminated or truncated
        if done:
            obs_dict, _ = env.reset()
        obs = self._process_obs(self._extract(obs_dict))
        info_out = (
            {"terminated": terminated, "truncated": truncated, **info}
            if isinstance(info, dict)
            else {"terminated": terminated, "truncated": truncated}
        )
        return obs, float(reward), float(done), info_out

    # ------------------------------------------------------------------
    # Public API
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
