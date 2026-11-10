from __future__ import annotations

import copy

import numpy as np
from scipy.integrate import solve_ivp

from norman.physics import model
from norman.physics.params import ReactorParams


class PythonReactor:
    backend = "python"

    def __init__(self, params: ReactorParams | None = None, rtol: float = 1e-8, atol: float = 1e-10):
        self.params = params or ReactorParams()
        self.params.validate()
        self.rtol = rtol
        self.atol = atol
        self.y = model.equilibrium_state(self.params, 1.0)
        self.rod = 0.25
        self.boron = 0.0
        self.bias = 0.0
        self.time = 0.0
        self.peak_power = 1.0
        self.reset(1.0, 0.25, 0.0)

    def reset(self, power: float, rod: float, boron: float) -> None:
        self.y = model.equilibrium_state(self.params, power)
        self.rod = float(np.clip(rod, 0.0, 1.0))
        self.boron = float(np.clip(boron, 0.0, self.params.boron_max))
        self.bias = model.critical_bias(self.params, self.y, self.rod, self.boron)
        self.time = 0.0
        self.peak_power = float(power)

    def set_state(self, y, rod: float, boron: float, bias: float, time: float = 0.0) -> None:
        self.y = np.array(y, dtype=float)
        self.rod, self.boron, self.bias, self.time = float(rod), float(boron), float(bias), float(time)

    @property
    def state(self) -> np.ndarray:
        return self.y.copy()

    @property
    def power(self) -> float:
        return float(self.y[model.N])

    def reactivity(self) -> float:
        return model.reactivity(self.params, self.y, self.rod, self.boron, self.bias)

    def reactivity_components(self) -> dict:
        return model.reactivity_components(self.params, self.y, self.rod, self.boron, self.bias)

    def derivatives(self, y, rod: float, boron: float) -> np.ndarray:
        return model.derivatives(self.params, np.asarray(y, dtype=float), rod, boron, self.bias)

    def copy(self) -> "PythonReactor":
        return copy.deepcopy(self)

    def _segments(self, duration: float, rod_rate: float, boron_rate: float):
        cuts = {0.0, duration}
        p = self.params
        for value, rate, upper in ((self.rod, rod_rate, 1.0), (self.boron, boron_rate, p.boron_max)):
            if rate > 0.0:
                t_hit = (upper - value) / rate
            elif rate < 0.0:
                t_hit = -value / rate
            else:
                continue
            if 0.0 < t_hit < duration:
                cuts.add(t_hit)
        edges = sorted(cuts)
        return list(zip(edges[:-1], edges[1:]))

    def advance(self, duration: float, rod_command: float, boron_command: float) -> None:
        p = self.params
        rod_rate = -float(np.clip(rod_command, -1.0, 1.0)) * p.rod_speed_max
        boron_rate = -float(np.clip(boron_command, -1.0, 1.0)) * p.boron_rate_max
        rod0, boron0 = self.rod, self.boron

        def inputs(t):
            rod = min(max(rod0 + rod_rate * t, 0.0), 1.0)
            boron = min(max(boron0 + boron_rate * t, 0.0), p.boron_max)
            return rod, boron

        def f(t, y):
            rod, boron = inputs(t)
            return model.derivatives(p, y, rod, boron, self.bias)

        def jac(t, y):
            rod, boron = inputs(t)
            return model.jacobian(p, y, rod, boron, self.bias)

        y = self.y
        peak = float(y[model.N])
        for t0, t1 in self._segments(duration, rod_rate, boron_rate):
            sol = solve_ivp(
                f, (t0, t1), y, method="Radau", jac=jac,
                rtol=self.rtol, atol=self.atol, dense_output=True,
            )
            if not sol.success:
                raise RuntimeError(f"Radau integration failed: {sol.message}")
            grid = np.linspace(t0, t1, max(2, int((t1 - t0) / p.substep) + 1))
            peak = max(peak, float(np.max(sol.sol(grid)[model.N])))
            y = sol.y[:, -1]

        self.y = np.array(y)
        self.rod, self.boron = inputs(duration)
        self.time += duration
        self.peak_power = peak
