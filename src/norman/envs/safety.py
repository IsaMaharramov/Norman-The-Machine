from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    overpower: float = 1.05
    trip_power: float = 1.09
    ramp_rate: float = 0.05
    fuel_temperature: float = 690.0
    axial_offset_band: float = 0.05
    steam_pressure_low: float = 5.5
    steam_pressure_high: float = 8.0


def check(limits: SafetyLimits, *, peak_power: float, power: float, previous_power: float,
          minutes: float, fuel_temperature: float, delta_i_error: float | None = None,
          steam_pressure: float | None = None) -> dict:
    ramp = abs(power - previous_power) / minutes
    out = {
        "overpower": peak_power > limits.overpower,
        "ramp_rate": ramp > limits.ramp_rate,
        "fuel_temperature": fuel_temperature > limits.fuel_temperature,
    }
    if delta_i_error is not None:
        out["axial_offset"] = abs(delta_i_error) > limits.axial_offset_band
    if steam_pressure is not None:
        out["steam_pressure"] = not limits.steam_pressure_low <= steam_pressure <= limits.steam_pressure_high
    return out


def is_trip(limits: SafetyLimits, peak_power: float) -> bool:
    return peak_power > limits.trip_power
