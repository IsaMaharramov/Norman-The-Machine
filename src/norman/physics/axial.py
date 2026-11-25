from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

from norman.physics.params import ReactorParams

try:
    from norman import _core
except ImportError:
    _core = None


@dataclass
class AxialParams:
    nodes: int = 40
    height: float = 366.0
    extrapolation: float = 8.0
    migration_area: float = 50.0
    design_axial_offset: float = -0.03
    max_shape_iterations: int = 200
    shape_tolerance: float = 1e-11
    poison_substep: float = 300.0

    def validate(self) -> None:
        if self.nodes < 4 or self.nodes % 2:
            raise ValueError("nodes must be an even number >= 4")
        if self.height <= 0 or self.extrapolation < 0 or self.migration_area <= 0 or self.poison_substep <= 0:
            raise ValueError("axial geometry parameters must be positive")

    def replace(self, **changes) -> AxialParams:
        return dataclasses.replace(self, **changes)


def leakage_matrix(a: AxialParams) -> tuple[np.ndarray, np.ndarray]:
    h = a.height / a.nodes
    inner = a.migration_area / h**2
    edge = a.migration_area / (h * (0.5 * h + a.extrapolation))
    left = np.full(a.nodes, inner)
    right = np.full(a.nodes, inner)
    left[0] = edge
    right[-1] = edge
    return 1.0 + left + right, np.full(a.nodes - 1, -inner)


def half_balance(values: np.ndarray) -> float:
    half = len(values) // 2
    bottom, top = values[:half].sum(), values[half:].sum()
    return float((top - bottom) / (top + bottom))


class PythonAxialCore:
    backend = "python"

    def __init__(self, params: ReactorParams | None = None, axial: AxialParams | None = None):
        self.params = params or ReactorParams()
        self.axial = axial or AxialParams()
        self.params.validate()
        self.axial.validate()
        n = self.axial.nodes
        self.node_height = self.axial.height / n
        self.a_diag, self.a_off = leakage_matrix(self.axial)
        self.k_infinity_reference = float(
            eigh_tridiagonal(self.a_diag, self.a_off, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
        shape = np.sin(math.pi * (np.arange(n) + 0.5) / n)
        self.shape = shape / shape.mean()
        p = self.params
        self.fuel_temperature = np.full(n, p.t_fuel_nominal)
        self.coolant_temperature = np.full(n, p.t_coolant_nominal)
        self.iodine = np.ones(n)
        self.xenon = np.ones(n)
        self.promethium = np.ones(n)
        self.samarium = np.ones(n)
        self.local_reactivity = np.zeros(n)
        self.inlet = p.t_inlet
        self.bias = 0.0
        self.tilt = 0.0
        self.reactivity = 0.0

    @property
    def nodes(self) -> int:
        return self.axial.nodes

    @property
    def k_eff(self) -> float:
        return 1.0 / (1.0 - self.reactivity)

    @property
    def axial_offset(self) -> float:
        return half_balance(self.shape)

    @property
    def xenon_axial_offset(self) -> float:
        return half_balance(self.xenon)

    @property
    def peaking(self) -> float:
        return float(self.shape.max())

    def solve_eigen(self, rho) -> tuple[float, np.ndarray]:
        rho = np.asarray(rho, dtype=float)
        f = self.k_infinity_reference / (1.0 - rho)
        d = self.a_diag / f
        e = self.a_off / np.sqrt(f[:-1] * f[1:])
        values, vectors = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
        phi = vectors[:, 0] / np.sqrt(f)
        phi = phi * np.sign(phi.sum())
        return float(values[0]), phi / phi.mean()

    def rod_coverage(self, rod: float) -> np.ndarray:
        tip = self.axial.height * (1.0 - min(max(rod, 0.0), 1.0))
        edges = (np.arange(self.nodes) + 1) * self.node_height
        return np.clip((edges - tip) / self.node_height, 0.0, 1.0)

    def _temperatures(self, coolant_power: float, fuel_power: float) -> None:
        p = self.params
        flow = p.coolant_flow_capacity
        conductance = p.fuel_to_coolant_conductance / self.nodes
        node = self.shape * p.rated_power / self.nodes
        t_out = self.inlet + np.cumsum(coolant_power * node) / flow
        t_in = np.concatenate([[self.inlet], t_out[:-1]])
        self.coolant_temperature = 0.5 * (t_in + t_out)
        self.fuel_temperature = self.coolant_temperature + p.fuel_power_fraction * fuel_power * node / conductance

    def _equilibrium_poisons(self, power: float) -> None:
        p = self.params
        u = power * self.shape
        self.iodine = u.copy()
        self.xenon = u * (p.lambda_xenon + p.xenon_burn_rate) / (p.lambda_xenon + p.xenon_burn_rate * u)
        self.promethium = u.copy()
        self.samarium = np.where(u > 0.0, 1.0, self.samarium)

    def _local(self, rod: float, boron: float) -> None:
        self.local_reactivity = self._compute_local(rod, boron)

    def rod_worth_curve(self, points: int, boron: float) -> np.ndarray:
        if points < 2:
            raise ValueError("a rod worth curve needs at least two points")
        rho = np.array([1.0 - self.solve_eigen(self._compute_local(z, boron))[0]
                        for z in np.linspace(0.0, 1.0, points)])
        return rho - rho[0]

    def _compute_local(self, rod: float, boron: float) -> np.ndarray:
        p = self.params
        z = (np.arange(self.nodes) + 0.5) / self.nodes
        return (
            self.bias + self.tilt * (z - 0.5) - p.rod_bank_worth * self.rod_coverage(rod)
            + p.boron_worth * boron
            + p.alpha_fuel * (self.fuel_temperature - p.t_fuel_nominal)
            + p.alpha_moderator * (self.coolant_temperature - p.t_coolant_nominal)
            - p.xenon_worth_full_power * self.xenon
            - p.samarium_worth_full_power * self.samarium
        )

    def _iterate(self, power: float, rod: float, boron: float, equilibrium: bool, hold_critical: bool,
                 coolant_power: float | None = None, fuel_power: float | None = None) -> float:
        coolant_power = power if coolant_power is None else coolant_power
        fuel_power = power if fuel_power is None else fuel_power
        a = self.axial
        for _ in range(a.max_shape_iterations):
            self._temperatures(coolant_power, fuel_power)
            if equilibrium:
                self._equilibrium_poisons(power)
            self._local(rod, boron)
            lam, nxt = self.solve_eigen(self.local_reactivity)
            rho = 1.0 - lam
            change = float(np.max(np.abs(nxt - self.shape)))
            self.shape = 0.3 * self.shape + 0.7 * nxt
            if hold_critical:
                self.bias -= rho
            if change < a.shape_tolerance and (not hold_critical or abs(rho) < a.shape_tolerance):
                break
        self._temperatures(coolant_power, fuel_power)
        if equilibrium:
            self._equilibrium_poisons(power)
        self._local(rod, boron)
        lam, self.shape = self.solve_eigen(self.local_reactivity)
        return 1.0 - lam

    def initialize(self, power: float, rod: float, boron: float) -> None:
        if power <= 0.0:
            raise ValueError("initial power must be positive")
        target = self.axial.design_axial_offset
        self.bias = 0.0

        def offset_error(tilt: float) -> float:
            self.tilt = tilt
            self._iterate(1.0, rod, boron, True, True)
            return self.axial_offset - target

        t0, t1 = 0.0, 0.01
        f0, f1 = offset_error(t0), offset_error(t1)
        for _ in range(40):
            if abs(f1) <= 1e-10 or f1 == f0:
                break
            t0, t1, f0 = t1, t1 - f1 * (t1 - t0) / (f1 - f0), f1
            f1 = offset_error(t1)
        self.reactivity = self._iterate(power, rod, boron, True, True)

    def update(self, power: float, rod: float, boron: float, coolant_power: float | None = None,
               fuel_power: float | None = None, inlet: float | None = None) -> float:
        self.inlet = self.params.t_inlet if inlet is None else float(inlet)
        self.reactivity = self._iterate(power, rod, boron, False, False, coolant_power, fuel_power)
        return self.reactivity

    def advance_poisons(self, dt: float, mean_power: float) -> None:
        p = self.params
        burn, burn_sm = p.xenon_burn_rate, p.samarium_burn_rate
        for i in range(self.nodes):
            u = mean_power * self.shape[i]
            m = np.zeros((5, 5))
            m[0, 0], m[0, 4] = -p.lambda_iodine, p.lambda_iodine * u
            m[1, 0] = p.xenon_iodine_source
            m[1, 1] = -(p.lambda_xenon + burn * u)
            m[1, 4] = p.xenon_direct_source * u
            m[2, 2], m[2, 4] = -p.lambda_promethium, p.lambda_promethium * u
            m[3, 2], m[3, 3] = burn_sm, -burn_sm * u
            y = np.array([self.iodine[i], self.xenon[i], self.promethium[i], self.samarium[i], 1.0])
            y = expm(m * dt) @ y
            self.iodine[i], self.xenon[i], self.promethium[i], self.samarium[i] = y[:4]

    def set_poisons(self, iodine, xenon, promethium, samarium) -> None:
        self.iodine, self.xenon = np.array(iodine, float), np.array(xenon, float)
        self.promethium, self.samarium = np.array(promethium, float), np.array(samarium, float)

    def copy(self) -> PythonAxialCore:
        return copy.deepcopy(self)


class CppAxialCore:
    backend = "cpp"
    ARRAYS = ("shape", "fuel_temperature", "coolant_temperature", "iodine", "xenon", "promethium",
              "samarium", "local_reactivity")
    SCALARS = ("nodes", "node_height", "k_infinity_reference", "reactivity", "k_eff", "bias", "tilt",
               "axial_offset", "xenon_axial_offset", "peaking", "inlet")

    def __init__(self, params: ReactorParams | None = None, axial: AxialParams | None = None):
        if _core is None:
            raise ImportError("the compiled extension norman._core is not built")
        from norman.physics.backend import native_params

        self.params = params or ReactorParams()
        self.axial = axial or AxialParams()
        self.params.validate()
        self.axial.validate()
        native = _core.AxialParams()
        for name in AxialParams.__dataclass_fields__:
            setattr(native, name, getattr(self.axial, name))
        self._c = _core.AxialCore(native_params(self.params), native)

    def __getattr__(self, name):
        if name in CppAxialCore.ARRAYS:
            return np.asarray(getattr(self._c, name))
        if name in CppAxialCore.SCALARS:
            return getattr(self._c, name)
        raise AttributeError(name)

    def initialize(self, power: float, rod: float, boron: float) -> None:
        self._c.initialize(float(power), float(rod), float(boron))

    def update(self, power: float, rod: float, boron: float, coolant_power: float | None = None,
               fuel_power: float | None = None, inlet: float | None = None) -> float:
        cp = -1.0 if coolant_power is None else float(coolant_power)
        fp = -1.0 if fuel_power is None else float(fuel_power)
        ti = -1.0 if inlet is None else float(inlet)
        return self._c.update(float(power), float(rod), float(boron), cp, fp, ti)

    def advance_poisons(self, dt: float, mean_power: float) -> None:
        self._c.advance_poisons(float(dt), float(mean_power))

    def solve_eigen(self, rho) -> tuple[float, np.ndarray]:
        lam, shape = self._c.solve_eigen(list(map(float, rho)))
        return lam, np.asarray(shape)

    def rod_coverage(self, rod: float) -> np.ndarray:
        return np.asarray(self._c.rod_coverage(float(rod)))

    def rod_worth_curve(self, points: int, boron: float) -> np.ndarray:
        return np.asarray(self._c.rod_worth_curve(int(points), float(boron)))

    def set_poisons(self, iodine, xenon, promethium, samarium) -> None:
        self._c.set_poisons(*(list(map(float, v)) for v in (iodine, xenon, promethium, samarium)))

    def copy(self) -> CppAxialCore:
        clone = object.__new__(CppAxialCore)
        clone.params, clone.axial, clone._c = self.params, self.axial, self._c.copy()
        return clone


def make_axial_core(params: ReactorParams | None = None, axial: AxialParams | None = None,
                    backend: str = "auto"):
    if backend == "python" or (backend == "auto" and _core is None):
        return PythonAxialCore(params, axial)
    return CppAxialCore(params, axial)
