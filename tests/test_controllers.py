import numpy as np
import pytest

from norman.control.pid import PIDController
from norman.control.shield import SafetyShield
from norman.envs.load_follow import LoadFollowEnv
from norman.evaluation import run_episode


@pytest.mark.parametrize("cycle", ["BOC", "MOC", "EOC"])
def test_pid_follows_the_eur_daily_cycle(cycle):
    env = LoadFollowEnv(scenario="eur_daily", cycle=cycle)
    _, m = run_episode(env, PIDController(), seed=0)
    assert not m["trip"]
    assert m["cost_steps"] == 0
    assert m["rmse_pct"] < 0.5


class Reckless:
    def reset(self):
        pass

    def act(self, obs):
        return np.array([1.0, 1.0], dtype=np.float32)


def test_reckless_controller_trips_without_shield():
    _, m = run_episode(LoadFollowEnv(hours=3.0), Reckless(), seed=0)
    assert m["trip"]


def test_shield_prevents_the_trip():
    env = SafetyShield(LoadFollowEnv(hours=3.0))
    _, m = run_episode(env, Reckless(), seed=0)
    assert not m["trip"]
    assert m["shield_interventions"] > 0
    assert m["rmse_pct"] < 1.0


def test_sac_agent_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from norman.agents.sac import ReplayBuffer, SACAgent, SACConfig

    env = LoadFollowEnv(hours=0.5)
    obs, _ = env.reset(seed=0)
    agent = SACAgent(env.observation_space.shape[0], 2, SACConfig(batch_size=8, hidden=32), device="cpu")
    buf = ReplayBuffer(env.observation_space.shape[0], 2, 100)
    rng = np.random.default_rng(0)
    for _ in range(16):
        a = agent.act(obs, deterministic=False)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
        nxt, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, nxt, float(term))
        obs = env.reset(seed=1)[0] if term or trunc else nxt
    stats = agent.update(buf, rng)
    assert np.isfinite(stats["critic_loss"])

    path = tmp_path / "agent.pt"
    agent.save(str(path))
    loaded = SACAgent.load(str(path), device="cpu")
    with torch.no_grad():
        assert np.allclose(agent.act(obs), loaded.act(obs))
