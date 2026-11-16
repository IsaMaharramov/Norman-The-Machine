from __future__ import annotations

import math

import numpy as np

from norman.physics import model
from norman.physics.axial import AxialParams, make_axial_core
from norman.physics.backend import make_reactor
from norman.physics.params import ReactorParams

ROD_TABLE_POINTS = 21


class AxialPlant:
    def __init__(self, params: ReactorParams | None = None, axial: AxialParams | None = None,
                 backend: str = "auto", coupling_interval: float = 5.0):
        if coupling_interval <= 0.0:
            raise ValueError("coupling_interval must be positive")
        self.point = make_reactor(params, backend)
        self.params = self.point.params
        self.core = make_axial_core(self.params, axial, backend)
        self.axial = self.core.axial
        self.backend = self.point.backend
        self.coupling_interval = coupling_interval
        self.base_bias = 0.0
        self.shape_correction = 0.0
        self._peak_power = 1.0

    def reset(self, power: float, rod: float, boron: float) -> None:
        self.point.reset(power, rod, boron)
        self.base_bias = self.point.bias
        self.shape_correction = 0.0
        self.core.initialize(power, self.point.rod, self.point.boron)
        self._install_rod_worth()
        self._peak_power = self.point.power

    def thermal_powers(self) -> tuple[float, float]:
        p, y = self.params, self.point.state
        coolant = (y[model.TC] - p.t_inlet) / (p.t_coolant_nominal - p.t_inlet)
        fuel = (y[model.TF] - y[model.TC]) / (p.t_fuel_nominal - p.t_coolant_nominal)
        return float(coolant), float(fuel)

    def advance(self, duration: float, rod_command: float, boron_command: float) -> None:
        self._install_rod_worth()
        steps = max(1, math.ceil(duration / self.coupling_interval - 1e-9))
        h = duration / steps
        peak = self.point.power
        for _ in range(steps):
            start = self.point.power
            self.point.advance(h, rod_command, boron_command)
            peak = max(peak, self.point.peak_power)
            self.core.advance_poisons(h, 0.5 * (start + self.point.power))
            coolant, fuel = self.thermal_powers()
            self.core.update(self.point.power, self.point.rod, self.point.boron, coolant, fuel)
            self._synchronize()
        self._peak_power = peak

    def _install_rod_worth(self) -> None:
        self.point.set_rod_worth_table(self.core.rod_worth_curve(ROD_TABLE_POINTS, self.point.boron))
        self._synchronize()

    def _synchronize(self) -> None:
        rho_point = self.point.reactivity() - self.shape_correction
        self.shape_correction = self.core.reactivity - rho_point
        self.point.set_state(self.point.state, self.point.rod, self.point.boron,
                             self.base_bias + self.shape_correction, self.point.time)

    def set_state(self, y, rod: float, boron: float, bias: float, time: float = 0.0) -> None:
        self.point.set_state(y, rod, boron, bias, time)

    @property
    def state(self) -> np.ndarray:
        return self.point.state

    @property
    def y(self) -> np.ndarray:
        return self.point.state

    power = property(lambda self: self.point.power)
    rod = property(lambda self: self.point.rod)
    boron = property(lambda self: self.point.boron)
    bias = property(lambda self: self.point.bias)
    time = property(lambda self: self.point.time)
    peak_power = property(lambda self: self._peak_power)
    axial_offset = property(lambda self: self.core.axial_offset)
    xenon_axial_offset = property(lambda self: self.core.xenon_axial_offset)
    peaking = property(lambda self: self.core.peaking)
    shape = property(lambda self: self.core.shape)

    @property
    def delta_i(self) -> float:
        return self.core.axial_offset * self.point.power

    @property
    def delta_i_target(self) -> float:
        return self.axial.design_axial_offset * self.point.power

    def reactivity(self) -> float:
        return self.point.reactivity()

    def reactivity_components(self) -> dict:
        out = self.point.reactivity_components()
        out["bias"] = self.base_bias
        out["shape"] = self.shape_correction
        return out

    def derivatives(self, y, rod: float, boron: float) -> np.ndarray:
        return self.point.derivatives(y, rod, boron)

    def copy(self) -> AxialPlant:
        clone = object.__new__(AxialPlant)
        clone.__dict__.update(self.__dict__)
        clone.point = self.point.copy()
        clone.core = self.core.copy()
        return clone
