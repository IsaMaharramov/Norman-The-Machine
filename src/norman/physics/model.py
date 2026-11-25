from __future__ import annotations

import math

import numpy as np

from norman.physics.params import ReactorParams
from norman.physics.water import saturation_pressure, saturation_pressure_slope

N, C0, TF, TC, IODINE, XENON, PM, SM = 0, 1, 7, 8, 9, 10, 11, 12
TH, TP, TCL, TS = 13, 14, 15, 16
STATE_SIZE = 17
STATE_NAMES = (
    "power", "c1", "c2", "c3", "c4", "c5", "c6", "t_fuel", "t_coolant",
    "iodine", "xenon", "promethium", "samarium",
    "t_hot_leg", "t_sg_primary", "t_cold_leg", "t_saturation",
)


def rod_worth(p: ReactorParams, rod: float) -> float:
    z = min(max(rod, 0.0), 1.0)
    table = p.rod_worth_table
    if len(table) >= 2:
        x = z * (len(table) - 1)
        k = min(int(x), len(table) - 2)
        w = x - k
        return (1.0 - w) * table[k] + w * table[k + 1]
    return -p.rod_bank_worth * (z - math.sin(2.0 * math.pi * z) / (2.0 * math.pi))


def inlet_temperature(p: ReactorParams, y: np.ndarray) -> float:
    return float(y[TCL]) if p.secondary else p.t_inlet


def steam_fraction(p: ReactorParams, t_saturation: float, load: float) -> float:
    available = p.turbine_valve_max * saturation_pressure(t_saturation) / p.steam_pressure_full
    return min(load, available)


def steam_fraction_slope(p: ReactorParams, t_saturation: float, load: float) -> float:
    available = p.turbine_valve_max * saturation_pressure(t_saturation) / p.steam_pressure_full
    if load <= available:
        return 0.0
    return p.turbine_valve_max * saturation_pressure_slope(t_saturation) / p.steam_pressure_full


def reactivity_components(p: ReactorParams, y: np.ndarray, rod: float, boron: float, bias: float) -> dict:
    return {
        "bias": bias,
        "rods": rod_worth(p, rod),
        "boron": p.boron_worth * boron,
        "doppler": p.alpha_fuel * (y[TF] - p.t_fuel_nominal),
        "moderator": p.alpha_moderator * (y[TC] - p.t_coolant_nominal),
        "xenon": -p.xenon_worth_full_power * y[XENON],
        "samarium": -p.samarium_worth_full_power * y[SM],
    }


def reactivity(p: ReactorParams, y: np.ndarray, rod: float, boron: float, bias: float) -> float:
    return sum(reactivity_components(p, y, rod, boron, bias).values())


def derivatives(p: ReactorParams, y: np.ndarray, rod: float, boron: float, bias: float,
                load: float = 1.0) -> np.ndarray:
    betas, lambdas = p.betas, p.lambdas
    lam = p.generation_time
    n = y[N]
    c = y[C0:C0 + 6]
    rho = reactivity(p, y, rod, boron, bias)

    dy = np.zeros(STATE_SIZE)
    dy[N] = (rho - p.beta) / lam * n + float(np.dot(betas, c)) / lam + p.neutron_source
    dy[C0:C0 + 6] = lambdas * (n - c)

    ha = p.fuel_to_coolant_conductance
    wc = p.coolant_flow_capacity
    q = p.rated_power * n
    t_in = inlet_temperature(p, y)
    dy[TF] = (p.fuel_power_fraction * q - ha * (y[TF] - y[TC])) / p.fuel_heat_capacity
    dy[TC] = (
        (1.0 - p.fuel_power_fraction) * q
        + ha * (y[TF] - y[TC])
        - 2.0 * wc * (y[TC] - t_in)
    ) / p.coolant_heat_capacity

    dy[IODINE] = p.lambda_iodine * (n - y[IODINE])
    dy[XENON] = (
        p.xenon_direct_source * n
        + p.xenon_iodine_source * y[IODINE]
        - (p.lambda_xenon + p.xenon_burn_rate * n) * y[XENON]
    )
    dy[PM] = p.lambda_promethium * (n - y[PM])
    dy[SM] = p.samarium_burn_rate * (y[PM] - n * y[SM])

    if p.secondary:
        ua = p.sg_conductance
        transfer = ua * (y[TP] - y[TS])
        dy[TH] = (2.0 * y[TC] - y[TCL] - y[TH]) / p.hot_leg_time
        dy[TP] = (wc * (y[TH] - y[TP]) - transfer) / p.sg_primary_capacity
        dy[TCL] = (y[TP] - y[TCL]) / p.cold_leg_time
        dy[TS] = (transfer - p.rated_power * steam_fraction(p, y[TS], load)) / p.sg_secondary_capacity
    return dy


def jacobian(p: ReactorParams, y: np.ndarray, rod: float, boron: float, bias: float,
             load: float = 1.0) -> np.ndarray:
    betas, lambdas = p.betas, p.lambdas
    lam = p.generation_time
    n = y[N]
    rho = reactivity(p, y, rod, boron, bias)
    J = np.zeros((STATE_SIZE, STATE_SIZE))

    J[N, N] = (rho - p.beta) / lam
    J[N, C0:C0 + 6] = betas / lam
    J[N, TF] = p.alpha_fuel * n / lam
    J[N, TC] = p.alpha_moderator * n / lam
    J[N, XENON] = -p.xenon_worth_full_power * n / lam
    J[N, SM] = -p.samarium_worth_full_power * n / lam

    for i in range(6):
        J[C0 + i, N] = lambdas[i]
        J[C0 + i, C0 + i] = -lambdas[i]

    ha = p.fuel_to_coolant_conductance
    wc = p.coolant_flow_capacity
    cf, cc = p.fuel_heat_capacity, p.coolant_heat_capacity
    J[TF, N] = p.fuel_power_fraction * p.rated_power / cf
    J[TF, TF] = -ha / cf
    J[TF, TC] = ha / cf
    J[TC, N] = (1.0 - p.fuel_power_fraction) * p.rated_power / cc
    J[TC, TF] = ha / cc
    J[TC, TC] = -(ha + 2.0 * wc) / cc

    J[IODINE, N] = p.lambda_iodine
    J[IODINE, IODINE] = -p.lambda_iodine
    J[XENON, N] = p.xenon_direct_source - p.xenon_burn_rate * y[XENON]
    J[XENON, IODINE] = p.xenon_iodine_source
    J[XENON, XENON] = -(p.lambda_xenon + p.xenon_burn_rate * n)
    J[PM, N] = p.lambda_promethium
    J[PM, PM] = -p.lambda_promethium
    J[SM, N] = -p.samarium_burn_rate * y[SM]
    J[SM, PM] = p.samarium_burn_rate
    J[SM, SM] = -p.samarium_burn_rate * n

    if p.secondary:
        ua = p.sg_conductance
        cp, cs = p.sg_primary_capacity, p.sg_secondary_capacity
        J[TC, TCL] = 2.0 * wc / cc
        J[TH, TC] = 2.0 / p.hot_leg_time
        J[TH, TCL] = -1.0 / p.hot_leg_time
        J[TH, TH] = -1.0 / p.hot_leg_time
        J[TP, TH] = wc / cp
        J[TP, TP] = -(wc + ua) / cp
        J[TP, TS] = ua / cp
        J[TCL, TP] = 1.0 / p.cold_leg_time
        J[TCL, TCL] = -1.0 / p.cold_leg_time
        J[TS, TP] = ua / cs
        J[TS, TS] = -(ua + p.rated_power * steam_fraction_slope(p, y[TS], load)) / cs
    return J


def equilibrium_state(p: ReactorParams, power: float) -> np.ndarray:
    if power <= 0.0:
        raise ValueError("equilibrium requires a positive power level")
    y = np.empty(STATE_SIZE)
    y[N] = power
    y[C0:C0 + 6] = power
    t_c = p.tavg_reference(power)
    t_cold = t_c - p.rated_power * power / (2.0 * p.coolant_flow_capacity)
    y[TC] = t_c
    y[TF] = t_c + p.fuel_power_fraction * p.rated_power * power / p.fuel_to_coolant_conductance
    y[IODINE] = power
    burn = p.xenon_burn_rate
    y[XENON] = power * (p.lambda_xenon + burn) / (p.lambda_xenon + burn * power)
    y[PM] = power
    y[SM] = 1.0
    y[TH] = 2.0 * t_c - t_cold
    y[TP] = t_cold
    y[TCL] = t_cold
    y[TS] = t_cold - p.rated_power * power / p.sg_conductance
    return y


def critical_bias(p: ReactorParams, y: np.ndarray, rod: float, boron: float) -> float:
    return -reactivity(p, y, rod, boron, 0.0) - p.generation_time * p.neutron_source / y[N]


def inhour_reactivity(p: ReactorParams, omega: float) -> float:
    return omega * p.generation_time + float(np.sum(p.betas * omega / (omega + p.lambdas)))


def inhour_root(p: ReactorParams, rho: float) -> float:
    if not 0.0 < rho < p.beta:
        raise ValueError("inhour_root expects 0 < rho < beta")
    lo, hi = 0.0, 1.0
    while inhour_reactivity(p, hi) < rho:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if inhour_reactivity(p, mid) < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
