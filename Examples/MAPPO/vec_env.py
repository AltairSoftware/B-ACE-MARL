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
from b_ace_py.red_team_policy import RedTeamPolicy


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

    # combat_area values are in NM; observations are normalized by 3000 GDM.
    # NM -> normalized: NM * NM2GDM / 3000 = NM * (1852/100) / 3000
    _NM_TO_NORM = 1852.0 / 100.0 / 3000.0

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

        # Convert NM boundaries to normalized obs coordinates
        k = self._NM_TO_NORM
        x_min = ca["x_min"] * k
        x_max = ca["x_max"] * k
        z_min = ca["z_min"] * k
        z_max = ca["z_max"] * k

        area_w = x_max - x_min
        area_d = z_max - z_min

        boundary = np.column_stack([
            (x - x_min) / area_w,   # dist to x_min (left)
            (x_max - x) / area_w,   # dist to x_max (right)
            (z - z_min) / area_d,   # dist to z_min (near)
            (z_max - z) / area_d,   # dist to z_max (far)
        ])                           # shape (n_agents, 4)

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


class RedTeamVecEnv:
    """
    Wraps PettingZooVecEnv to handle a rule-based red team transparently.

    From the PPO algorithm's perspective, only blue agents exist (n_agents = n_blue).
    Red agents are controlled internally by RedTeamPolicy on every step.

    Output shapes (n_envs=E, n_blue=A, obs_dim=O):
        reset() -> obs:     (E, A, O)
        step()  -> obs:     (E, A, O)
                   rewards: (E,)
                   dones:   (E,)
                   infos:   list[dict]  length E
    """

    def __init__(
        self,
        env_fns: list[Callable[[], B_ACE_GodotPettingZooWrapper]],
        red_team_policy: RedTeamPolicy,
        combat_area: dict | None = None,
    ):
        self._inner = PettingZooVecEnv(env_fns, combat_area=combat_area)

        # n_blue is reported by Godot's env_info; read from the first live env.
        n_blue = self._inner.envs[0].n_blue
        self._n_blue          = n_blue
        self._n_red           = self._inner.n_agents - n_blue
        self._red_agent_names = self._inner.agents[n_blue:]

        # Inject obs_maps into the policy using the first inner env's map.
        ref_env = self._inner.envs[0]
        red_team_policy.obs_maps = {
            name: ref_env.obs_map[name] for name in self._red_agent_names
        }
        self._red_policy = red_team_policy

        # Last observed red-agent obs per environment, shape (n_envs, n_red, obs_dim)
        self._last_red_obs: np.ndarray | None = None

        # RL-visible dimensions (only blue agents exposed to the algorithm)
        self.n_envs         = self._inner.n_envs
        self.n_agents       = n_blue
        self.action_dim     = self._inner.action_dim
        self.obs_dim        = self._inner.obs_dim
        self.global_obs_dim = self.obs_dim * n_blue
        self.agents         = self._inner.agents[:n_blue]

    # ------------------------------------------------------------------
    # Public API (mirrors PettingZooVecEnv)
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset all envs, cache red obs, return only blue obs (E, A, O)."""
        all_obs = self._inner.reset()                         # (E, n_all, O)
        self._last_red_obs = all_obs[:, self._n_blue:, :]    # (E, n_red, O)
        return all_obs[:, :self._n_blue, :]                  # (E, n_blue, O)

    def step(
        self, blue_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step all environments with combined blue + red actions.

        blue_actions: (n_envs, n_blue, action_dim)  — from the RL algorithm.
        Returns blue obs, rewards, dones, infos (red agents excluded from outputs).
        """
        # Compute red actions using last cached observations
        red_actions = self._compute_red_actions()             # (n_envs, n_red, action_dim)
        all_actions = np.concatenate([blue_actions, red_actions], axis=1)  # (n_envs, n_all, D)

        all_obs, rewards, dones, infos = self._inner.step(all_actions)

        self._last_red_obs = all_obs[:, self._n_blue:, :]    # cache for next step
        blue_obs = all_obs[:, :self._n_blue, :]
        return blue_obs, rewards, dones, infos

    def get_global_obs(self, obs: np.ndarray) -> np.ndarray:
        """obs (E, n_blue, O) -> global_obs (E, n_blue*O)."""
        return obs.reshape(obs.shape[0], -1)

    def close(self) -> None:
        self._inner.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_red_actions(self) -> np.ndarray:
        """
        Run RedTeamPolicy for every env and every red agent.

        Returns shape (n_envs, n_red, action_dim).
        """
        n_envs, n_red, obs_dim = self._last_red_obs.shape
        actions = np.zeros((n_envs, n_red, self.action_dim), dtype=np.float32)
        for e in range(n_envs):
            actions[e] = self._red_policy.act_batch(
                self._red_agent_names, self._last_red_obs[e]
            )
        return actions
