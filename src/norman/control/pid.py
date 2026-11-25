from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from norman.envs.load_follow import OBS


@dataclass
class PIDConfig:
    kp: float = 0.6
    ki: float = 0.08
    feedforward: float = 1.0
    deadband: float = 0.002
    integral_limit: float = 0.05
    power_coefficient_pcm: float = 1300.0
    rod_worth_pcm: float = 2500.0
    rod_speed: float = 1.0 / 180.0
    interval_s: float = 60.0
    rod_reference: float = 0.25
    rod_band: float = 0.03
    boron_gain: float = 8.0


class PIDController:
    name = "pid"

    def __init__(self, config: PIDConfig | None = None, boron_control: bool = True):
        self.cfg = config or PIDConfig()
        self.boron_control = boron_control
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        c = self.cfg
        power = float(obs[OBS["power"]])
        target = float(obs[OBS["target_next"]])
        target_15 = float(obs[OBS["target_15min"]])
        rod = float(obs[OBS["rod_insertion"]])

        error = target - power
        if abs(error) > c.deadband:
            self.integral = float(np.clip(self.integral + error, -c.integral_limit, c.integral_limit))
        trend = (target_15 - target) / 14.0

        demand = c.kp * error + c.ki * self.integral + c.feedforward * trend
        if abs(error) <= c.deadband and abs(trend) < 1e-4:
            demand = 0.0
        d_rho = c.power_coefficient_pcm * demand
        differential = c.rod_worth_pcm * (1.0 - math.cos(2.0 * math.pi * rod))
        differential = max(differential, 0.05 * c.rod_worth_pcm)
        d_withdraw = d_rho / differential
        rod_cmd = float(np.clip(d_withdraw / (c.rod_speed * c.interval_s), -1.0, 1.0))

        boron_cmd = 0.0
        offset = rod - c.rod_reference
        if self.boron_control and abs(offset) > c.rod_band:
            boron_cmd = float(np.clip(-c.boron_gain * (offset - math.copysign(c.rod_band, offset)), -1.0, 1.0))
        return np.array([rod_cmd, boron_cmd], dtype=np.float32)


def boron_for_rod_band(rod: float, reference: float, band: float, gain: float) -> float:
    offset = rod - reference
    if abs(offset) <= band:
        return 0.0
    return float(np.clip(-gain * (offset - math.copysign(band, offset)), -1.0, 1.0))


def rod_command(d_rho_pcm: float, rod: float, rod_worth_pcm: float, rod_speed: float, interval_s: float) -> float:
    differential = rod_worth_pcm * (1.0 - math.cos(2.0 * math.pi * rod))
    differential = max(differential, 0.05 * rod_worth_pcm)
    return float(np.clip(d_rho_pcm / differential / (rod_speed * interval_s), -1.0, 1.0))


@dataclass
class TavgConfig:
    tavg_gain: float = 0.5
    mismatch_gain: float = 0.5
    temperature_coefficient_pcm: float = 27.5
    power_coefficient_pcm: float = 1300.0
    deadband_k: float = 0.8
    nominal_rise_k: float = 16.0
    rod_worth_pcm: float = 2500.0
    rod_speed: float = 1.0 / 180.0
    interval_s: float = 60.0
    rod_reference: float = 0.25
    rod_band: float = 0.03
    boron_gain: float = 8.0


class TavgController:
    name = "tavg"

    def __init__(self, config: TavgConfig | None = None, boron_control: bool = True):
        self.cfg = config or TavgConfig()
        self.boron_control = boron_control

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        c = self.cfg
        power = float(obs[OBS["power"]])
        load = float(obs[OBS["target_next"]])
        rod = float(obs[OBS["rod_insertion"]])
        tavg_dev = 10.0 * float(obs[OBS["coolant_temp_dev_10K"]])
        error = c.nominal_rise_k * (load - 1.0) - tavg_dev
        if abs(error) < c.deadband_k:
            error = 0.0
        d_rho = (c.tavg_gain * c.temperature_coefficient_pcm * error
                 + c.mismatch_gain * c.power_coefficient_pcm * (load - power))
        rod_cmd = rod_command(d_rho, rod, c.rod_worth_pcm, c.rod_speed, c.interval_s)
        boron_cmd = 0.0
        if self.boron_control:
            boron_cmd = boron_for_rod_band(rod, c.rod_reference, c.rod_band, c.boron_gain)
        return np.array([rod_cmd, boron_cmd], dtype=np.float32)


class HoldController:
    name = "hold"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)
