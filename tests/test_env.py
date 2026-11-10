import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from norman.envs.load_follow import OBS, LoadFollowEnv
from norman.envs.scenarios import SCENARIOS, make_profile


def test_registered_env_passes_gymnasium_checks():
    env = gym.make("Norman/LoadFollow-v1", hours=1.0)
    check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_profiles_stay_in_operating_band(name):
    profile = make_profile(name, np.random.default_rng(0), 1440)
    assert profile.shape[0] >= 1441
    assert profile.min() >= 0.3 and profile.max() <= 1.0
    assert np.max(np.abs(np.diff(profile))) <= 0.05 + 1e-9


def test_same_seed_same_episode():
    def rollout(seed):
        env = LoadFollowEnv(scenario="random", cycle="random", hours=2.0, observation_noise=0.01)
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for _ in range(30):
            obs, *_ = env.step(np.zeros(2))
            out.append(obs)
        return np.array(out)

    assert np.array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_peek_does_not_change_the_plant():
    env = LoadFollowEnv(hours=1.0)
    env.reset(seed=0)
    before = env.reactor.state
    prediction = env.peek(np.array([1.0, 1.0]))
    assert np.array_equal(env.reactor.state, before)
    obs, *_ = env.step(np.array([1.0, 1.0]))
    assert prediction["power"] == pytest.approx(float(obs[OBS["power"]]))


def test_runaway_withdrawal_trips_and_ends_episode():
    env = LoadFollowEnv(hours=2.0)
    env.reset(seed=0)
    terminated, reward, info = False, 0.0, {}
    for _ in range(env.steps):
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
        if terminated or truncated:
            break
    assert terminated and info["trip"]
    assert reward < -1.0
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))


def test_rods_only_mode_has_one_action():
    env = LoadFollowEnv(boron_control=False, hours=1.0)
    assert env.action_space.shape == (1,)
    env.reset(seed=0)
    boron = env.reactor.boron
    env.step(np.array([0.5]))
    assert env.reactor.boron == boron


def test_python_backend_runs():
    env = LoadFollowEnv(hours=0.25, backend="python")
    env.reset(seed=0)
    for _ in range(env.steps):
        env.step(np.zeros(2))
    assert env.reactor.backend == "python"
