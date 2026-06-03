"""
MAPPO PPO algorithm internals.

Users do NOT need to modify this file.
To customize reward shaping, edit reward_fn() in mappo_b_ace.py.

Public interface:
    train(cfg, b_ace_config, reward_fn=None, restore_path=None)
"""
import time
import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from networks import MAPPOActor, MAPPOCritic
from vec_env import PettingZooVecEnv, RedTeamVecEnv, make_env as _make_single_env
from b_ace_py.red_team_policy import RedTeamPolicy


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    next_done: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and discounted returns for one environment."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - float(next_done)
            next_val = next_value
        else:
            next_nonterminal = 1.0 - float(dones[t + 1])
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO minibatch update
# ---------------------------------------------------------------------------

def ppo_update(
    actor: MAPPOActor,
    critic: MAPPOCritic,
    optimizer: torch.optim.Adam,
    b_local_obs: torch.Tensor,
    b_global_obs: torch.Tensor,
    b_actions: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    b_values: torch.Tensor,
    algo_cfg: dict,
    n_agents: int,
    N: int,
) -> dict:
    """
    Run update_epochs x minibatches of clipped PPO.

    N = T * n_envs (total rollout samples).
    Returns a dict of mean loss statistics.
    """
    clip_coef = algo_cfg["clip_coef"]
    vf_coef   = algo_cfg["vf_coef"]
    ent_coef  = algo_cfg["ent_coef"]
    max_grad  = algo_cfg["max_grad_norm"]
    n_epochs  = algo_cfg["update_epochs"]
    mb_size   = (N * n_agents) // algo_cfg["num_minibatches"]

    stats: dict[str, list] = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "approx_kl": [], "clipfrac": [],
    }

    for _ in range(n_epochs):
        perm = torch.randperm(N * n_agents)
        for start in range(0, N * n_agents, mb_size):
            mb      = perm[start : start + mb_size]
            mb_step = mb // n_agents

            new_logprob, entropy = actor.get_log_prob_entropy(
                b_local_obs[mb], b_actions[mb]
            )
            new_value = critic(b_global_obs[mb_step]).squeeze(-1)

            logratio = new_logprob - b_logprobs[mb]
            ratio    = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac  = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            mb_adv = b_advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
            ).mean()

            mb_ret = b_returns[mb_step]
            mb_val = b_values[mb_step]
            if algo_cfg["clip_vloss"]:
                v_clipped = mb_val + (new_value - mb_val).clamp(-clip_coef, clip_coef)
                vf_loss   = torch.max(
                    (new_value - mb_ret).pow(2),
                    (v_clipped - mb_ret).pow(2),
                ).mean()
            else:
                vf_loss = (new_value - mb_ret).pow(2).mean()

            ent_loss = entropy.mean()
            loss = pg_loss - ent_coef * ent_loss + vf_coef * vf_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), max_grad
            )
            optimizer.step()

            stats["policy_loss"].append(pg_loss.item())
            stats["value_loss"].append(vf_loss.item())
            stats["entropy"].append(ent_loss.item())
            stats["approx_kl"].append(approx_kl)
            stats["clipfrac"].append(clipfrac)

    with torch.no_grad():
        ev = 1.0 - (
            np.var(b_returns.numpy() - b_values.numpy()) /
            (np.var(b_returns.numpy()) + 1e-8)
        )

    return {k: float(np.mean(v)) for k, v in stats.items()} | {"explained_var": float(ev)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    cfg: dict,
    b_ace_config: dict,
    reward_fn: Callable[[np.ndarray, float, bool], float] | None = None,
    restore_path: str | None = None,
    red_team_policy: "RedTeamPolicy | None" = None,
) -> None:
    """
    Run MAPPO training.

    Args:
        cfg:          Full config dict loaded from config.yaml.
        b_ace_config: B-ACE wrapper config built by build_b_ace_config().
        reward_fn:    Optional reward shaping function with signature
                      (obs, reward, done) -> float.
                      If None, the raw simulation reward is used as-is.
        restore_path: Path to a checkpoint .pt file to resume from.
    """
    algo    = cfg["algo"]
    log_cfg = cfg["logging"]
    n_envs  = algo.get("num_envs", 1)
    T       = algo["num_steps"]

    print("=" * 60)
    print(f"B-ACE MAPPO  exp={log_cfg['exp_name']}  n_envs={n_envs}")
    print("=" * 60)
    print(f"  lr={algo['learning_rate']}  steps={T}  "
          f"total={algo['total_timesteps']:,}  epochs={algo['update_epochs']}")
    if reward_fn is not None:
        print(f"  reward_fn: {reward_fn.__name__}")
    print()

    # ── Environment ───────────────────────────────────────────────────────
    combat_area = cfg.get("combat_area")
    env_fns = [lambda idx=i: _make_single_env(b_ace_config, idx) for i in range(n_envs)]

    if red_team_policy is not None:
        # RedTeamVecEnv starts the inner envs, reads n_blue from Godot env_info,
        # and auto-injects obs_maps into the policy — no second process needed.
        venv = RedTeamVecEnv(env_fns, red_team_policy, combat_area=combat_area)
        print(f"  red_team_policy: chase+fire  n_blue={venv.n_agents}  n_red={venv._n_red}")
    else:
        venv = PettingZooVecEnv(env_fns, combat_area=combat_area)

    if combat_area:
        print(f"  combat_area: x=[{combat_area['x_min']}, {combat_area['x_max']}]"
              f"  z=[{combat_area['z_min']}, {combat_area['z_max']}]"
              f"  penalty={combat_area.get('out_penalty', -1.0)}")

    obs_dim        = venv.obs_dim
    action_dim     = venv.action_dim
    n_agents       = venv.n_agents
    global_obs_dim = venv.global_obs_dim

    print(f"obs_dim={obs_dim}  action_dim={action_dim}  "
          f"n_agents={n_agents}  global_obs_dim={global_obs_dim}")

    # ── Networks & optimizer ──────────────────────────────────────────────
    hidden = algo["hidden_sizes"]
    actor  = MAPPOActor(obs_dim, action_dim, hidden)
    critic = MAPPOCritic(global_obs_dim, hidden)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=algo["learning_rate"], eps=1e-5,
    )

    start_iteration = 1
    global_step     = 0

    if restore_path:
        ckpt = torch.load(restore_path, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt["iteration"] + 1
        global_step     = ckpt["global_step"]
        print(f"Resumed from '{restore_path}' "
              f"(iter={start_iteration}, step={global_step:,})")

    # ── Logging ───────────────────────────────────────────────────────────
    run_name = (
        f"{log_cfg['exp_name']}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = Path(log_cfg["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(save_dir))

    # ── Rollout buffers  shape: (T, n_envs, ...) ─────────────────────────
    b_local_obs  = np.zeros((T, n_envs, n_agents, obs_dim),    dtype=np.float32)
    b_global_obs = np.zeros((T, n_envs, global_obs_dim),        dtype=np.float32)
    b_actions    = np.zeros((T, n_envs, n_agents, action_dim), dtype=np.float32)
    b_logprobs   = np.zeros((T, n_envs, n_agents),              dtype=np.float32)
    b_rewards    = np.zeros((T, n_envs),                        dtype=np.float32)
    b_dones      = np.zeros((T, n_envs),                        dtype=np.float32)
    b_values     = np.zeros((T, n_envs),                        dtype=np.float32)

    total_timesteps = algo["total_timesteps"]
    num_iterations  = total_timesteps // (T * n_envs)
    ckpt_interval   = log_cfg["checkpoint_interval"]

    obs        = venv.reset()   # (n_envs, n_agents, obs_dim)
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lens    = np.zeros(n_envs, dtype=np.int32)
    ep_count   = 0
    start_time = time.time()

    print(f"Iterations: {num_iterations}  |  Steps/iter: {T * n_envs}  |  "
          f"Total: {total_timesteps:,}  |  Save: {save_dir}")
    print("-" * 60)

    for iteration in range(start_iteration, num_iterations + 1):

        # ── Rollout collection ────────────────────────────────────────────
        for step in range(T):
            obs_flat = torch.as_tensor(
                obs.reshape(n_envs * n_agents, obs_dim), dtype=torch.float32
            )
            global_obs_t = torch.as_tensor(
                venv.get_global_obs(obs), dtype=torch.float32
            )

            with torch.no_grad():
                actions_t, logprobs_t, _ = actor.get_action(obs_flat)
                values_t = critic(global_obs_t).squeeze(-1)

            actions_np = actions_t.numpy().reshape(n_envs, n_agents, action_dim)

            next_obs, rewards, dones, _ = venv.step(actions_np)

            # ── reward_fn hook ────────────────────────────────────────────
            if reward_fn is not None:
                rewards = np.array(
                    [reward_fn(next_obs[e], float(rewards[e]), bool(dones[e]))
                     for e in range(n_envs)],
                    dtype=np.float32,
                )

            b_local_obs[step]  = obs
            b_global_obs[step] = venv.get_global_obs(obs)
            b_actions[step]    = actions_np
            b_logprobs[step]   = logprobs_t.numpy().reshape(n_envs, n_agents)
            b_rewards[step]    = rewards
            b_dones[step]      = dones
            b_values[step]     = values_t.numpy()

            obs          = next_obs
            global_step += n_envs
            ep_rewards  += rewards
            ep_lens     += 1

            for e in range(n_envs):
                if dones[e]:
                    ep_count += 1
                    writer.add_scalar("charts/episodic_return", ep_rewards[e], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lens[e],    global_step)
                    ep_rewards[e] = 0.0
                    ep_lens[e]    = 0

        # ── Bootstrap value ───────────────────────────────────────────────
        with torch.no_grad():
            next_values = critic(
                torch.as_tensor(venv.get_global_obs(obs), dtype=torch.float32)
            ).squeeze(-1).numpy()
        next_dones = b_dones[-1]

        # ── GAE per environment ───────────────────────────────────────────
        advantages = np.zeros((T, n_envs), dtype=np.float32)
        returns    = np.zeros((T, n_envs), dtype=np.float32)
        for e in range(n_envs):
            advantages[:, e], returns[:, e] = compute_gae(
                b_rewards[:, e], b_values[:, e], b_dones[:, e],
                float(next_values[e]), bool(next_dones[e]),
                algo["gamma"], algo["gae_lambda"],
            )

        # ── Flatten  N = T*n_envs,  Na = N*n_agents ──────────────────────
        N  = T * n_envs
        Na = N * n_agents
        flat_obs      = torch.as_tensor(b_local_obs.reshape(Na, obs_dim))
        flat_global   = torch.as_tensor(b_global_obs.reshape(N, global_obs_dim))
        flat_actions  = torch.as_tensor(b_actions.reshape(Na, action_dim))
        flat_logprobs = torch.as_tensor(b_logprobs.reshape(Na))
        flat_adv      = torch.as_tensor(
            np.repeat(advantages.reshape(N), n_agents).astype(np.float32)
        )
        flat_returns  = torch.as_tensor(returns.reshape(N))
        flat_values   = torch.as_tensor(b_values.reshape(N))

        # ── PPO update ────────────────────────────────────────────────────
        stats = ppo_update(
            actor, critic, optimizer,
            flat_obs, flat_global, flat_actions, flat_logprobs,
            flat_adv, flat_returns, flat_values,
            algo, n_agents, N,
        )

        sps = int(global_step / (time.time() - start_time))
        print(
            f"iter={iteration:5d}  "
            f"pol={stats['policy_loss']:+.4f}  "
            f"val={stats['value_loss']:.4f}  "
            f"ent={stats['entropy']:.4f}  "
            f"kl={stats['approx_kl']:.4f}  "
            f"ev={stats['explained_var']:.3f}  "
            f"SPS={sps}"
        )
        writer.add_scalar("charts/SPS",               sps,                    global_step)
        writer.add_scalar("losses/policy_loss",        stats["policy_loss"],   global_step)
        writer.add_scalar("losses/value_loss",         stats["value_loss"],    global_step)
        writer.add_scalar("losses/entropy",            stats["entropy"],       global_step)
        writer.add_scalar("losses/approx_kl",          stats["approx_kl"],    global_step)
        writer.add_scalar("losses/clipfrac",           stats["clipfrac"],      global_step)
        writer.add_scalar("losses/explained_variance", stats["explained_var"], global_step)

        # ── Checkpoint ────────────────────────────────────────────────────
        if iteration % ckpt_interval == 0:
            ckpt_path = save_dir / f"checkpoint_{iteration}.pt"
            torch.save({
                "actor_state_dict":     actor.state_dict(),
                "critic_state_dict":    critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration":            iteration,
                "global_step":          global_step,
                "config":               cfg,
            }, ckpt_path)
            print(f"  [ckpt] saved -> {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────
    writer.close()
    venv.close()
    final_path = save_dir / "final.pt"
    torch.save({
        "actor_state_dict":  actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "iteration":         num_iterations,
        "global_step":       global_step,
        "config":            cfg,
    }, final_path)
    print(f"\nTraining complete. Final model -> {final_path}")
