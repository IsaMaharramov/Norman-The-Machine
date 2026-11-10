import gymnasium as gym
import numpy as np

import norman
from norman.control.pid import PIDController


class ProportionalRods:
    def __init__(self, gain: float = 2.0):
        self.gain = gain

    def reset(self):
        pass

    def act(self, obs):
        error = obs[1] - obs[0]
        return np.array([np.clip(self.gain * error, -1.0, 1.0), 0.0], dtype=np.float32)


def rollout(controller, scenario: str) -> float:
    env = gym.make("Norman/LoadFollow-v1", scenario=scenario, cycle="MOC")
    obs, _ = env.reset(seed=0)
    controller.reset()
    errors, cost, done = [], 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller.act(obs))
        errors.append(info["error"])
        cost += info["cost"]
        done = terminated or truncated
    rmse = float(np.sqrt(np.mean(np.square(errors))) * 100)
    print(f"{type(controller).__name__:<18} {scenario:<12} rmse={rmse:5.2f}%  cost steps={cost:.0f}  trip={info['trip']}")
    return rmse


if __name__ == "__main__":
    print(f"norman {norman.__version__}")
    for scenario in ("eur_daily", "fast_ramp"):
        rollout(PIDController(), scenario)
        rollout(ProportionalRods(), scenario)
