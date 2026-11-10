from __future__ import annotations

import gymnasium as gym
import numpy as np

from norman.control.pid import PIDController


def _severity(prediction: dict) -> float:
    return 10.0 * prediction["trip"] + sum(prediction["violations"].values())


class SafetyShield(gym.Wrapper):
    def __init__(self, env: gym.Env, fallback=None):
        super().__init__(env)
        self.fallback = fallback or PIDController(boron_control=env.unwrapped.boron_control)
        self.interventions = 0
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.fallback.reset()
        self.interventions = 0
        self._obs = obs
        return obs, info

    def choose(self, action, fallback_action) -> tuple[np.ndarray, bool]:
        plant = self.env.unwrapped
        dims = plant.action_space.shape[0]
        proposed = np.asarray(action, dtype=np.float32)
        prediction = plant.peek(proposed)
        if _severity(prediction) == 0:
            return proposed, False
        candidates = [proposed * s for s in (0.5, 0.25, 0.1)]
        candidates += [np.asarray(fallback_action, dtype=np.float32)[:dims], np.zeros(dims, dtype=np.float32)]
        best, best_score = proposed, _severity(prediction)
        for candidate in candidates:
            score = _severity(plant.peek(candidate))
            if score < best_score:
                best, best_score = candidate, score
            if score == 0:
                break
        return best, best is not proposed

    def step(self, action):
        fallback_action = self.fallback.act(self._obs)
        chosen, intervened = self.choose(action, fallback_action)
        self.interventions += int(intervened)
        obs, reward, terminated, truncated, info = self.env.step(chosen)
        info["shield_intervened"] = intervened
        info["shield_interventions"] = self.interventions
        self._obs = obs
        return obs, reward, terminated, truncated, info
