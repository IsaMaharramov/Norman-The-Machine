from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass
class SACConfig:
    hidden: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1
    init_alpha: float = 0.2


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs: torch.Tensor):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - 2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mu)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]

    def __len__(self) -> int:
        return self.size


class SACAgent:
    name = "sac"

    def __init__(self, obs_dim: int, act_dim: int, config: SACConfig | None = None, device: str | None = None):
        self.cfg = config or SACConfig()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        c = self.cfg
        self.actor = Actor(obs_dim, act_dim, c.hidden).to(self.device)
        self.critic = TwinCritic(obs_dim, act_dim, c.hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(c.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.lr)
        self.deterministic = True

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp())

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool | None = None) -> np.ndarray:
        det = self.deterministic if deterministic is None else deterministic
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action = self.actor.sample(o)
        out = mean_action if det else action
        return out.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict:
        c = self.cfg
        o, a, r, o2, d = (torch.as_tensor(x, device=self.device) for x in buffer.sample(c.batch_size, rng))
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(o2)
            q1_t, q2_t = self.critic_target(o2, a2)
            target = r + c.gamma * (1.0 - d) * (torch.min(q1_t, q2_t) - alpha * logp2)
        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        self.critic.requires_grad_(False)
        a_pi, logp, _ = self.actor.sample(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                tp.mul_(1.0 - c.tau).add_(c.tau * p)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": self.alpha,
            "entropy": float(-logp.mean()),
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "config": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": float(self.log_alpha),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> SACAgent:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], SACConfig(**ckpt["config"]), device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            agent.log_alpha.fill_(ckpt["log_alpha"])
        agent.extra = ckpt.get("extra", {})
        return agent
