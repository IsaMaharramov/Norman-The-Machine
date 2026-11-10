from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from norman.envs import safety
from norman.envs.safety import SafetyLimits
from norman.envs.scenarios import make_profile
from norman.physics import model
from norman.physics.backend import make_reactor
from norman.physics.params import CYCLE_INITIAL_BORON, ReactorParams, cycle_params, sample_cycle

PREVIEW_MINUTES = (1, 15, 30, 60)
OBS_NAMES = (
    "power", "target_next", "target_15min", "target_30min", "target_60min",
    "rod_insertion", "boron_kppm", "fuel_temp_dev_100K", "coolant_temp_dev_10K",
    "xenon", "iodine", "reactivity_dollars", "power_rate_5pct_min", "episode_progress",
)
OBS = {name: i for i, name in enumerate(OBS_NAMES)}


class LoadFollowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "eur_daily",
        cycle: str = "MOC",
        hours: float = 24.0,
        control_interval: float = 60.0,
        boron_control: bool = True,
        randomize: bool = False,
        observation_noise: float = 0.0,
        backend: str = "auto",
        params: ReactorParams | None = None,
        limits: SafetyLimits | None = None,
        initial_rod: float = 0.25,
        initial_boron: float | None = None,
        error_scale: float = 0.05,
        actuator_penalty: float = 0.005,
        reward_gamma: float = 0.99,
    ):
        super().__init__()
        self.scenario = scenario
        self.cycle = cycle
        self.control_interval = float(control_interval)
        self.steps = int(round(hours * 3600.0 / self.control_interval))
        self.minutes_per_step = self.control_interval / 60.0
        self.boron_control = boron_control
        self.randomize = randomize
        self.observation_noise = observation_noise
        self.backend = backend
        self.base_params = params
        self.limits = limits or SafetyLimits()
        self.initial_rod = initial_rod
        self.initial_boron = initial_boron
        self.error_scale = error_scale
        self.actuator_penalty = actuator_penalty
        self.reward_gamma = reward_gamma

        n_act = 2 if boron_control else 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)

        self.reactor = None
        self.profile = np.ones(self.steps + 61)
        self.t = 0
        self.history: list[dict] = []
        self.cycle_name = cycle
        self._prev_power = 1.0
        self._tripped = False

    def _build_reactor(self):
        rng = self.np_random
        boron = self.initial_boron
        if self.randomize or self.cycle.lower() == "random":
            drawn = sample_cycle(rng)
            params, self.cycle_name = drawn.params, drawn.name
            boron = drawn.initial_boron if boron is None else boron
        else:
            self.cycle_name = self.cycle.upper()
            params = self.base_params or cycle_params(self.cycle_name)
            if boron is None:
                boron = CYCLE_INITIAL_BORON.get(self.cycle_name, 650.0)
        reactor = make_reactor(params, self.backend)
        return reactor, boron

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.reactor, boron = self._build_reactor()
        minutes = int(math.ceil(self.steps * self.minutes_per_step))
        profile_min = make_profile(options.get("scenario", self.scenario), self.np_random, minutes,
                                   preview=max(PREVIEW_MINUTES) + 1)
        step_times = np.arange(self.steps + max(PREVIEW_MINUTES) + 1) * self.minutes_per_step
        self.profile = np.interp(step_times, np.arange(len(profile_min)), profile_min)
        self.reactor.reset(float(self.profile[0]), self.initial_rod, boron)
        self.t = 0
        self._prev_power = self.reactor.power
        self._tripped = False
        self.history = [self._snapshot(action=np.zeros(2), reward=0.0, cost=0.0, violations={})]
        return self._observation(), self._info(0.0, {}, False)

    def target(self, offset_steps: int = 0) -> float:
        idx = min(self.t + offset_steps, len(self.profile) - 1)
        return float(self.profile[idx])

    def _preview(self, minutes: int) -> float:
        return self.target(max(1, int(round(minutes / self.minutes_per_step))))

    def _observation(self) -> np.ndarray:
        r = self.reactor
        p = r.params
        y = r.state
        measured = y[model.N]
        if self.observation_noise > 0.0:
            measured += self.np_random.normal(0.0, self.observation_noise)
        rate = (y[model.N] - self._prev_power) / self.minutes_per_step
        obs = np.array([
            measured,
            self._preview(1),
            self._preview(15),
            self._preview(30),
            self._preview(60),
            r.rod,
            r.boron / 1000.0,
            (y[model.TF] - p.t_fuel_nominal) / 100.0,
            (y[model.TC] - p.t_coolant_nominal) / 10.0,
            y[model.XENON],
            y[model.IODINE],
            r.reactivity() / p.beta,
            rate / self.limits.ramp_rate,
            self.t / self.steps,
        ], dtype=np.float32)
        return obs

    def _commands(self, action) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        rod_cmd = float(a[0])
        boron_cmd = float(a[1]) if self.boron_control and a.size > 1 else 0.0
        return rod_cmd, boron_cmd

    def peek(self, action) -> dict:
        clone = self.reactor.copy()
        rod_cmd, boron_cmd = self._commands(action)
        clone.advance(self.control_interval, rod_cmd, boron_cmd)
        y = clone.state
        violations = safety.check(
            self.limits, peak_power=clone.peak_power, power=y[model.N],
            previous_power=self.reactor.power, minutes=self.minutes_per_step,
            fuel_temperature=y[model.TF],
        )
        return {
            "power": float(y[model.N]),
            "peak_power": float(clone.peak_power),
            "error": float(y[model.N] - self.target(1)),
            "violations": violations,
            "trip": safety.is_trip(self.limits, clone.peak_power),
        }

    def step(self, action):
        if self._tripped:
            raise RuntimeError("the reactor has tripped; call reset()")
        rod_cmd, boron_cmd = self._commands(action)
        self._prev_power = self.reactor.power
        self.reactor.advance(self.control_interval, rod_cmd, boron_cmd)
        self.t += 1

        y = self.reactor.state
        power = float(y[model.N])
        error = power - self.target()
        violations = safety.check(
            self.limits, peak_power=self.reactor.peak_power, power=power,
            previous_power=self._prev_power, minutes=self.minutes_per_step,
            fuel_temperature=float(y[model.TF]),
        )
        cost = 1.0 if any(violations.values()) else 0.0
        reward = -min((error / self.error_scale) ** 2, 1.0)
        reward -= self.actuator_penalty * (abs(rod_cmd) + abs(boron_cmd))

        trip = safety.is_trip(self.limits, self.reactor.peak_power)
        terminated = False
        if trip:
            self._tripped = True
            terminated = True
            remaining = self.steps - self.t + 1
            g = self.reward_gamma
            reward = -(1.0 - g ** remaining) / (1.0 - g)
            cost = 1.0
        truncated = self.t >= self.steps

        self.history.append(self._snapshot(np.array([rod_cmd, boron_cmd]), reward, cost, violations, trip))
        return self._observation(), float(reward), terminated, truncated, self._info(cost, violations, trip)

    def _snapshot(self, action, reward, cost, violations, trip=False) -> dict:
        r = self.reactor
        y = r.state
        rho = r.reactivity_components()
        return {
            "t_min": self.t * self.minutes_per_step,
            "power": float(y[model.N]),
            "peak_power": float(r.peak_power),
            "target": self.target(),
            "rod": float(r.rod),
            "boron": float(r.boron),
            "t_fuel": float(y[model.TF]),
            "t_coolant": float(y[model.TC]),
            "iodine": float(y[model.IODINE]),
            "xenon": float(y[model.XENON]),
            "samarium": float(y[model.SM]),
            "rho_pcm": {k: v * 1e5 for k, v in rho.items()},
            "rod_cmd": float(action[0]),
            "boron_cmd": float(action[1]) if len(action) > 1 else 0.0,
            "reward": float(reward),
            "cost": float(cost),
            "violations": {k: bool(v) for k, v in violations.items()},
            "trip": bool(trip),
        }

    def _info(self, cost, violations, trip) -> dict:
        r = self.reactor
        return {
            "cost": float(cost),
            "violations": {k: bool(v) for k, v in violations.items()},
            "trip": bool(trip),
            "power": float(r.power),
            "target": self.target(),
            "error": float(r.power - self.target()),
            "cycle": self.cycle_name,
            "rod": float(r.rod),
            "boron": float(r.boron),
        }
