from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from norman.envs import safety
from norman.envs.safety import SafetyLimits
from norman.envs.scenarios import make_profile
from norman.physics import model
from norman.physics.axial import AxialParams
from norman.physics.backend import make_reactor
from norman.physics.params import CYCLE_INITIAL_BORON, ReactorParams, cycle_params, sample_cycle
from norman.physics.plant import AxialPlant
from norman.physics.water import saturation_pressure

PREVIEW_MINUTES = (1, 15, 30, 60)
OBS_NAMES = (
    "power", "target_next", "target_15min", "target_30min", "target_60min",
    "rod_insertion", "boron_kppm", "fuel_temp_dev_100K", "coolant_temp_dev_10K",
    "xenon", "iodine", "reactivity_dollars", "power_rate_5pct_min", "episode_progress",
)
OBS = {name: i for i, name in enumerate(OBS_NAMES)}
AXIAL_OBS_NAMES = ("axial_offset_10pct", "delta_i_error_band", "xenon_axial_offset_10pct", "peaking_excess")
AXIAL_OBS = {name: len(OBS_NAMES) + i for i, name in enumerate(AXIAL_OBS_NAMES)}
SECONDARY_OBS_NAMES = ("electric_output", "tavg_error_5K", "steam_pressure_dev_MPa", "cold_leg_dev_10K")


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
        core: str = "point",
        axial: AxialParams | None = None,
        coupling_interval: float = 5.0,
        secondary: bool = False,
        tavg_weight: float = 0.2,
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
        if core not in ("point", "axial"):
            raise ValueError("core must be 'point' or 'axial'")
        self.core = core
        self.axial = axial or AxialParams()
        self.coupling_interval = coupling_interval
        self.secondary = secondary
        self.tavg_weight = tavg_weight

        n_act = 2 if boron_control else 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32)

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
        if self.secondary:
            params = params.replace(secondary=True)
        if self.core == "axial":
            return AxialPlant(params, self.axial, self.backend, self.coupling_interval), boron
        return make_reactor(params, self.backend), boron

    @property
    def observation_names(self) -> tuple:
        return (OBS_NAMES + (AXIAL_OBS_NAMES if self.core == "axial" else ())
                + (SECONDARY_OBS_NAMES if self.secondary else ()))

    def _electric(self, reactor) -> float:
        return float(reactor.steam_power) if self.secondary else float(reactor.power)

    def _steam_pressure(self, reactor) -> float | None:
        if not self.secondary:
            return None
        return saturation_pressure(float(reactor.state[model.TS]))

    def _tavg_error(self, reactor) -> float:
        return float(reactor.state[model.TC] - reactor.params.tavg_reference(reactor.load))

    def _delta_i_error(self, reactor) -> float | None:
        if self.core != "axial":
            return None
        return float(reactor.delta_i - reactor.delta_i_target)

    def _violations(self, reactor, previous_power: float) -> dict:
        y = reactor.state
        return safety.check(
            self.limits, peak_power=reactor.peak_power, power=float(y[model.N]),
            previous_power=previous_power, minutes=self.minutes_per_step,
            fuel_temperature=float(y[model.TF]), delta_i_error=self._delta_i_error(reactor),
            steam_pressure=self._steam_pressure(reactor),
        )

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
        if self.core == "axial":
            axial = np.array([
                r.axial_offset / 0.1,
                self._delta_i_error(r) / self.limits.axial_offset_band,
                r.xenon_axial_offset / 0.1,
                r.peaking - 1.0,
            ], dtype=np.float32)
            obs = np.concatenate([obs, axial])
        if self.secondary:
            plant = np.array([
                self._electric(r),
                self._tavg_error(r) / 5.0,
                self._steam_pressure(r) - p.steam_pressure_full,
                (r.inlet_temperature - p.t_inlet) / 10.0,
            ], dtype=np.float32)
            obs = np.concatenate([obs, plant])
        return obs

    def _commands(self, action) -> tuple[float, float]:
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        rod_cmd = float(a[0])
        boron_cmd = float(a[1]) if self.boron_control and a.size > 1 else 0.0
        return rod_cmd, boron_cmd

    def peek(self, action) -> dict:
        clone = self.reactor.copy()
        rod_cmd, boron_cmd = self._commands(action)
        clone.advance(self.control_interval, rod_cmd, boron_cmd, self._load(1))
        y = clone.state
        violations = self._violations(clone, self.reactor.power)
        return {
            "power": float(y[model.N]),
            "peak_power": float(clone.peak_power),
            "error": float(self._electric(clone) - self.target(1)),
            "violations": violations,
            "trip": safety.is_trip(self.limits, clone.peak_power),
        }

    def _load(self, offset: int) -> float | None:
        return self.target(offset) if self.secondary else None

    def step(self, action):
        if self._tripped:
            raise RuntimeError("the reactor has tripped; call reset()")
        rod_cmd, boron_cmd = self._commands(action)
        self._prev_power = self.reactor.power
        self.reactor.advance(self.control_interval, rod_cmd, boron_cmd, self._load(1))
        self.t += 1

        error = self._electric(self.reactor) - self.target()
        violations = self._violations(self.reactor, self._prev_power)
        cost = 1.0 if any(violations.values()) else 0.0
        reward = -min((error / self.error_scale) ** 2, 1.0)
        if self.secondary:
            reward -= self.tavg_weight * min((self._tavg_error(self.reactor) / 5.0) ** 2, 1.0)
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
        snap = {
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
        if self.core == "axial":
            snap.update({
                "axial_offset": float(r.axial_offset),
                "delta_i": float(r.delta_i),
                "delta_i_target": float(r.delta_i_target),
                "xenon_axial_offset": float(r.xenon_axial_offset),
                "peaking": float(r.peaking),
                "shape": [float(v) for v in r.shape],
                "xenon_profile": [float(v) for v in r.core.xenon],
            })
        if self.secondary:
            snap.update({
                "electric": self._electric(r),
                "tavg_error": self._tavg_error(r),
                "steam_pressure": self._steam_pressure(r),
                "t_cold": float(r.inlet_temperature),
                "t_hot": float(y[model.TH]),
            })
        return snap

    def _info(self, cost, violations, trip) -> dict:
        r = self.reactor
        return {
            "cost": float(cost),
            "violations": {k: bool(v) for k, v in violations.items()},
            "trip": bool(trip),
            "power": float(r.power),
            "target": self.target(),
            "error": float(self._electric(r) - self.target()),
            "cycle": self.cycle_name,
            "rod": float(r.rod),
            "boron": float(r.boron),
            **({"axial_offset": float(r.axial_offset), "delta_i_error": self._delta_i_error(r)}
               if self.core == "axial" else {}),
            **({"electric": self._electric(r), "tavg_error": self._tavg_error(r),
                "steam_pressure": self._steam_pressure(r)} if self.secondary else {}),
        }
