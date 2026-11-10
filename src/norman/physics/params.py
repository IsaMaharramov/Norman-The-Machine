from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

PCM = 1e-5

KEEPIN_U235_FRACTIONS = (0.033, 0.219, 0.196, 0.395, 0.115, 0.042)
KEEPIN_U235_DECAY = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)


@dataclass
class ReactorParams:
    beta: float = 0.0058
    group_fractions: tuple = KEEPIN_U235_FRACTIONS
    group_decay: tuple = KEEPIN_U235_DECAY
    generation_time: float = 2.0e-5
    neutron_source: float = 5.0e-7

    rated_power: float = 3.0e9
    fuel_power_fraction: float = 0.974
    t_inlet: float = 292.0
    t_coolant_nominal: float = 308.0
    t_fuel_nominal: float = 650.0
    fuel_heat_capacity: float = 4.27e7
    coolant_heat_capacity: float = 1.2e8

    alpha_fuel: float = -2.5 * PCM
    alpha_moderator: float = -25.0 * PCM
    boron_worth: float = -8.0 * PCM

    rod_bank_worth: float = 2500.0 * PCM
    rod_speed_max: float = 1.0 / 180.0
    boron_rate_max: float = 3.0 / 60.0
    boron_max: float = 2000.0

    nominal_flux: float = 3.0e13
    gamma_iodine: float = 0.0639
    gamma_xenon: float = 0.00237
    gamma_promethium: float = 0.0113
    lambda_iodine: float = 2.87e-5
    lambda_xenon: float = 2.09e-5
    lambda_promethium: float = 3.63e-6
    sigma_xenon: float = 2.65e-18
    sigma_samarium: float = 4.08e-20
    nu: float = 2.43

    substep: float = 0.25

    @property
    def betas(self) -> np.ndarray:
        return self.beta * np.asarray(self.group_fractions, dtype=float)

    @property
    def lambdas(self) -> np.ndarray:
        return np.asarray(self.group_decay, dtype=float)

    @property
    def coolant_flow_capacity(self) -> float:
        return self.rated_power / (2.0 * (self.t_coolant_nominal - self.t_inlet))

    @property
    def fuel_to_coolant_conductance(self) -> float:
        return (
            self.fuel_power_fraction
            * self.rated_power
            / (self.t_fuel_nominal - self.t_coolant_nominal)
        )

    @property
    def xenon_burn_rate(self) -> float:
        return self.sigma_xenon * self.nominal_flux

    @property
    def samarium_burn_rate(self) -> float:
        return self.sigma_samarium * self.nominal_flux

    @property
    def xenon_worth_full_power(self) -> float:
        gamma = self.gamma_iodine + self.gamma_xenon
        burn = self.xenon_burn_rate
        return gamma / self.nu * burn / (self.lambda_xenon + burn)

    @property
    def samarium_worth_full_power(self) -> float:
        return self.gamma_promethium / self.nu

    @property
    def xenon_direct_source(self) -> float:
        gamma = self.gamma_iodine + self.gamma_xenon
        return self.gamma_xenon / gamma * (self.lambda_xenon + self.xenon_burn_rate)

    @property
    def xenon_iodine_source(self) -> float:
        gamma = self.gamma_iodine + self.gamma_xenon
        return self.gamma_iodine / gamma * (self.lambda_xenon + self.xenon_burn_rate)

    def replace(self, **changes) -> ReactorParams:
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["group_fractions"] = list(self.group_fractions)
        out["group_decay"] = list(self.group_decay)
        return out

    def validate(self) -> None:
        if len(self.group_fractions) != 6 or len(self.group_decay) != 6:
            raise ValueError("six delayed-neutron groups are required")
        if not math.isclose(sum(self.group_fractions), 1.0, abs_tol=1e-6):
            raise ValueError("delayed-neutron group fractions must sum to 1")
        if self.t_fuel_nominal <= self.t_coolant_nominal <= self.t_inlet:
            raise ValueError("nominal temperatures must satisfy T_fuel > T_coolant > T_inlet")
        positive = (
            "beta", "generation_time", "rated_power", "fuel_heat_capacity",
            "coolant_heat_capacity", "rod_speed_max", "boron_rate_max", "substep",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


CYCLE_PRESETS = {
    "BOC": dict(
        beta=0.0065, generation_time=2.2e-5, alpha_fuel=-2.2 * PCM,
        alpha_moderator=-12.0 * PCM, boron_worth=-7.0 * PCM,
    ),
    "MOC": dict(
        beta=0.0058, generation_time=2.0e-5, alpha_fuel=-2.5 * PCM,
        alpha_moderator=-25.0 * PCM, boron_worth=-8.0 * PCM,
    ),
    "EOC": dict(
        beta=0.0052, generation_time=1.7e-5, alpha_fuel=-2.9 * PCM,
        alpha_moderator=-45.0 * PCM, boron_worth=-9.0 * PCM,
    ),
}

CYCLE_INITIAL_BORON = {"BOC": 1200.0, "MOC": 650.0, "EOC": 40.0}


@dataclass
class CycleState:
    name: str
    params: ReactorParams
    initial_boron: float
    initial_rod: float = 0.25
    extra: dict = field(default_factory=dict)


def cycle_params(name: str = "MOC", base: ReactorParams | None = None) -> ReactorParams:
    key = name.upper()
    if key not in CYCLE_PRESETS:
        raise ValueError(f"unknown cycle '{name}', expected one of {sorted(CYCLE_PRESETS)}")
    return (base or ReactorParams()).replace(**CYCLE_PRESETS[key])


def sample_cycle(rng: np.random.Generator, spread: float = 0.1) -> CycleState:
    name = str(rng.choice(sorted(CYCLE_PRESETS)))
    p = cycle_params(name)
    jitter = lambda v: float(v * (1.0 + rng.uniform(-spread, spread)))
    p = p.replace(
        alpha_fuel=jitter(p.alpha_fuel),
        alpha_moderator=jitter(p.alpha_moderator),
        boron_worth=jitter(p.boron_worth),
        rod_bank_worth=jitter(p.rod_bank_worth),
        generation_time=jitter(p.generation_time),
    )
    boron = jitter(CYCLE_INITIAL_BORON[name])
    return CycleState(name=name, params=p, initial_boron=boron)
