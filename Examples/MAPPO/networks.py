"""
MAPPO network definitions for B-ACE.

MAPPOActor  : decentralized, takes local obs  -> action distribution
MAPPOCritic : centralized,   takes global obs -> state value
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _build_mlp(input_dim: int, hidden_sizes: list[int]) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [_layer_init(nn.Linear(in_dim, h)), nn.Tanh()]
        in_dim = h
    return nn.Sequential(*layers), in_dim


class MAPPOActor(nn.Module):
    """
    Decentralized actor: pi(a | local_obs).

    Outputs a squashed Gaussian. log_std is a learnable parameter
    shared across the batch (not obs-dependent), which keeps the
    implementation simple and stable.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        trunk, last_dim = _build_mlp(obs_dim, hidden_sizes)
        self.trunk = trunk
        self.mu_head = _layer_init(nn.Linear(last_dim, action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        mu = self.mu_head(hidden)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action_tanh, log_prob, entropy)."""
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        # log prob with tanh correction: log pi(a) = log N(x) - log(1 - tanh^2(x))
        log_prob = (dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action with no sampling noise. Use for evaluation."""
        mu, _ = self(obs)
        return torch.tanh(mu)

    def get_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy of a given (pre-tanh-inverted) action."""
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        # Inverse tanh to recover the pre-squash sample x
        action_clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        x = torch.atanh(action_clipped)
        log_prob = (dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Centralized critic: V(global_obs).

    global_obs = concatenation of all blue agents' local observations.
    Only used during training (CTDE).
    """

    def __init__(self, global_obs_dim: int, hidden_sizes: list[int]):
        super().__init__()
        trunk, last_dim = _build_mlp(global_obs_dim, hidden_sizes)
        self.trunk = trunk
        self.value_head = _layer_init(nn.Linear(last_dim, 1), gain=1.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(global_obs))


# ---------------------------------------------------------------------------
# Quick self-test (run as: python networks.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OBS_DIM = 22
    ACTION_DIM = 4
    N_AGENTS = 1
    BATCH = 8
    HIDDEN = [256, 256]

    actor = MAPPOActor(OBS_DIM, ACTION_DIM, HIDDEN)
    critic = MAPPOCritic(OBS_DIM * N_AGENTS, HIDDEN)

    obs = torch.randn(BATCH, OBS_DIM)
    global_obs = torch.randn(BATCH, OBS_DIM * N_AGENTS)

    action, log_prob, entropy = actor.get_action(obs)
    det_action = actor.get_deterministic_action(obs)
    value = critic(global_obs)

    print(f"obs shape           : {obs.shape}")
    print(f"action shape        : {action.shape}       (expected: ({BATCH}, {ACTION_DIM}))")
    print(f"log_prob shape      : {log_prob.shape}      (expected: ({BATCH},))")
    print(f"entropy shape       : {entropy.shape}       (expected: ({BATCH},))")
    print(f"det_action shape    : {det_action.shape}   (expected: ({BATCH}, {ACTION_DIM}))")
    print(f"value shape         : {value.shape}       (expected: ({BATCH}, 1))")
    print(f"action range        : [{action.min():.3f}, {action.max():.3f}]  (should be in (-1, 1))")
    print(f"det_action range    : [{det_action.min():.3f}, {det_action.max():.3f}]  (should be in (-1, 1))")
    print("networks.py self-test passed.")
