from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    overpower: float = 1.05
    trip_power: float = 1.09
    ramp_rate: float = 0.05
    fuel_temperature: float = 690.0


def check(limits: SafetyLimits, *, peak_power: float, power: float, previous_power: float,
          minutes: float, fuel_temperature: float) -> dict:
    ramp = abs(power - previous_power) / minutes
    return {
        "overpower": peak_power > limits.overpower,
        "ramp_rate": ramp > limits.ramp_rate,
        "fuel_temperature": fuel_temperature > limits.fuel_temperature,
    }


def is_trip(limits: SafetyLimits, peak_power: float) -> bool:
    return peak_power > limits.trip_power
