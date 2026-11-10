from __future__ import annotations

import warnings

import numpy as np

from norman.physics.params import ReactorParams
from norman.physics.reference import PythonReactor

try:
    from norman import _core
except ImportError:
    _core = None

PARAM_FIELDS = tuple(ReactorParams.__dataclass_fields__)


def cpp_available() -> bool:
    return _core is not None


class CppReactor:
    backend = "cpp"

    def __init__(self, params: ReactorParams | None = None):
        if _core is None:
            raise ImportError("the compiled extension norman._core is not built")
        self.params = params or ReactorParams()
        self.params.validate()
        native = _core.Params()
        for name in PARAM_FIELDS:
            value = getattr(self.params, name)
            setattr(native, name, list(value) if isinstance(value, tuple) else value)
        self._r = _core.Reactor(native)

    def reset(self, power: float, rod: float, boron: float) -> None:
        self._r.reset(power, rod, boron)

    def set_state(self, y, rod: float, boron: float, bias: float, time: float = 0.0) -> None:
        self._r.set_state(list(map(float, y)), rod, boron, bias, time)

    def advance(self, duration: float, rod_command: float, boron_command: float) -> None:
        self._r.advance(float(duration), float(rod_command), float(boron_command))

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self._r.state, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.state

    power = property(lambda self: self._r.power)
    rod = property(lambda self: self._r.rod)
    boron = property(lambda self: self._r.boron)
    bias = property(lambda self: self._r.bias)
    time = property(lambda self: self._r.time)
    peak_power = property(lambda self: self._r.peak_power)

    def reactivity(self) -> float:
        return self._r.reactivity()

    def reactivity_components(self) -> dict:
        return dict(self._r.reactivity_components())

    def derivatives(self, y, rod: float, boron: float) -> np.ndarray:
        return np.asarray(self._r.derivatives(list(map(float, y)), rod, boron))

    def copy(self) -> "CppReactor":
        clone = object.__new__(CppReactor)
        clone.params = self.params
        clone._r = self._r.copy()
        return clone


def make_reactor(params: ReactorParams | None = None, backend: str = "auto"):
    if backend == "cpp":
        return CppReactor(params)
    if backend == "python":
        return PythonReactor(params)
    if backend != "auto":
        raise ValueError("backend must be 'auto', 'cpp' or 'python'")
    if _core is not None:
        return CppReactor(params)
    warnings.warn(
        "norman._core is not built; using the slower SciPy backend. "
        "Install with `pip install .` to compile the C++ core.",
        RuntimeWarning,
        stacklevel=2,
    )
    return PythonReactor(params)
