from __future__ import annotations

from typing import Callable

import numpy as np

ProfileFn = Callable[[np.random.Generator, int], np.ndarray]
SCENARIOS: dict[str, ProfileFn] = {}
DESCRIPTIONS: dict[str, str] = {}


def scenario(name: str, description: str):
    def register(fn: ProfileFn) -> ProfileFn:
        SCENARIOS[name] = fn
        DESCRIPTIONS[name] = description
        return fn

    return register


def _segments(points: list[tuple[float, float]], minutes: int) -> np.ndarray:
    times = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)
    return np.interp(np.arange(minutes + 1, dtype=float), times, values)


def rate_limit(profile: np.ndarray, max_rate_per_min: float) -> np.ndarray:
    out = profile.copy()
    for i in range(1, len(out)):
        step = np.clip(out[i] - out[i - 1], -max_rate_per_min, max_rate_per_min)
        out[i] = out[i - 1] + step
    return out


@scenario("eur_daily", "European Utility Requirements daily cycle 12-3-6-3: 100% -> 50% -> 100%")
def eur_daily(rng: np.random.Generator, minutes: int) -> np.ndarray:
    pts = [(0, 1.0), (720, 1.0), (900, 0.5), (1260, 0.5), (1440, 1.0), (max(minutes, 1440), 1.0)]
    return _segments(pts, minutes)


@scenario("fast_ramp", "5%/min ramps 100% -> 50% -> 100%, then hours of xenon transient")
def fast_ramp(rng: np.random.Generator, minutes: int) -> np.ndarray:
    pts = [(0, 1.0), (120, 1.0), (130, 0.5), (370, 0.5), (380, 1.0), (max(minutes, 1440), 1.0)]
    return _segments(pts, minutes)


@scenario("duck_curve", "Net load after solar and wind: midday dip, steep evening ramp")
def duck_curve(rng: np.random.Generator, minutes: int) -> np.ndarray:
    t = np.arange(minutes + 1) / 60.0
    hour = t % 24.0
    solar_share = rng.uniform(0.35, 0.5)
    solar = np.clip(np.sin(np.pi * (hour - 6.5) / 12.0), 0.0, None) ** 1.5
    evening = 0.06 * np.exp(-0.5 * ((hour - 19.0) / 1.2) ** 2)
    wind = np.zeros_like(t)
    for i in range(1, len(t)):
        wind[i] = wind[i - 1] - wind[i - 1] / 180.0 + rng.normal(0.0, 0.004)
    net = 1.0 - solar_share * solar + evening + wind
    net = np.clip(net, 0.5, 1.0)
    return rate_limit(net, 0.02)


@scenario("frequency_response", "Near-full-power operation with secondary frequency-control set-point changes")
def frequency_response(rng: np.random.Generator, minutes: int) -> np.ndarray:
    base = rng.uniform(0.9, 0.95)
    signal = np.empty(minutes + 1)
    level = 0.0
    i = 0
    while i <= minutes:
        hold = int(rng.integers(3, 15))
        level = float(np.clip(level + rng.normal(0.0, 0.025), -0.05, 0.05))
        signal[i:i + hold] = level
        i += hold
    return rate_limit(base + signal, 0.01)


@scenario("random", "Randomized holds and ramps between 50% and 100%, for training")
def random_profile(rng: np.random.Generator, minutes: int) -> np.ndarray:
    level = rng.uniform(0.5, 1.0)
    pts = [(0.0, level)]
    t = 0.0
    while t < minutes:
        t += rng.uniform(30.0, 240.0)
        pts.append((t, level))
        new_level = rng.uniform(0.5, 1.0)
        rate = rng.uniform(0.005, 0.05)
        t += abs(new_level - level) / rate
        level = new_level
        pts.append((t, level))
    return _segments(pts, minutes)


def make_profile(name: str, rng: np.random.Generator, minutes: int, preview: int = 60) -> np.ndarray:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}', expected one of {sorted(SCENARIOS)}")
    profile = np.asarray(SCENARIOS[name](rng, minutes), dtype=float)[: minutes + 1]
    return np.concatenate([profile, np.full(preview, profile[-1])])


BENCHMARK_SCENARIOS = ("eur_daily", "fast_ramp", "duck_curve", "frequency_response")
