from __future__ import annotations

import numpy as np

VIOLATION_TYPES = ("overpower", "ramp_rate", "fuel_temperature", "axial_offset", "steam_pressure")


def episode_metrics(history: list[dict], horizon_steps: int, minutes_per_step: float,
                    rated_power_mw: float = 3000.0, interventions: int = 0) -> dict:
    steps = history[1:]
    output = "electric" if "electric" in history[0] else "power"
    power = np.array([h[output] for h in steps])
    target = np.array([h["target"] for h in steps])
    tripped = any(h["trip"] for h in steps)

    missing = horizon_steps - len(steps)
    if missing > 0 and len(steps) > 0:
        power = np.concatenate([power, np.zeros(missing)])
        last_target = target[-1]
        target = np.concatenate([target, np.full(missing, last_target)])

    error = power - target
    hours_per_step = minutes_per_step / 60.0
    rods = np.array([h["rod"] for h in history])
    boron = np.array([h["boron"] for h in history])
    d_boron = np.diff(boron)

    axial = {}
    if "delta_i" in history[0]:
        dev = np.array([h["delta_i"] - h["delta_i_target"] for h in steps])
        axial = {
            "delta_i_rms_pct": float(np.sqrt(np.mean(dev ** 2)) * 100.0),
            "max_peaking": float(max(h["peaking"] for h in history)),
            "xenon_ao_swing": float(np.ptp([h["xenon_axial_offset"] for h in history])),
        }
    plant = {}
    if "tavg_error" in history[0]:
        tavg = np.array([h["tavg_error"] for h in steps])
        pressure = np.array([h["steam_pressure"] for h in steps])
        plant = {
            "tavg_rms_k": float(np.sqrt(np.mean(tavg ** 2))),
            "tavg_max_k": float(np.max(np.abs(tavg))),
            "steam_pressure_min": float(pressure.min()),
            "steam_pressure_max": float(pressure.max()),
        }
    return {
        "rmse_pct": float(np.sqrt(np.mean(error ** 2)) * 100.0),
        "mae_pct": float(np.mean(np.abs(error)) * 100.0),
        "max_abs_error_pct": float(np.max(np.abs(error)) * 100.0),
        "time_within_1pct": float(np.mean(np.abs(error) < 0.01)),
        "energy_deficit_mwh": float(np.sum(np.clip(-error, 0.0, None)) * hours_per_step * rated_power_mw),
        "rod_travel_strokes": float(np.sum(np.abs(np.diff(rods)))),
        "boron_dilution_ppm": float(-np.sum(d_boron[d_boron < 0])),
        "boration_ppm": float(np.sum(d_boron[d_boron > 0])),
        "cost_steps": int(sum(h["cost"] > 0 for h in steps)),
        **{f"viol_{k}": int(sum(h["violations"].get(k, False) for h in steps)) for k in VIOLATION_TYPES},
        "trip": bool(tripped),
        "steps_completed": len(steps),
        "shield_interventions": int(interventions),
        "return": float(sum(h["reward"] for h in steps)),
        **axial,
        **plant,
    }


def aggregate(rows: list[dict], keys=("rmse_pct", "max_abs_error_pct", "time_within_1pct",
                                      "energy_deficit_mwh", "rod_travel_strokes",
                                      "boron_dilution_ppm", "boration_ppm", "cost_steps",
                                      "shield_interventions")) -> dict:
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["trips"] = int(sum(r["trip"] for r in rows))
    out["episodes"] = len(rows)
    return out
