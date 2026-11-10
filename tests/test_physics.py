import math

import numpy as np
import pytest

from norman.physics import model
from norman.physics.backend import CppReactor, cpp_available, make_reactor
from norman.physics.params import ReactorParams
from norman.physics.reference import PythonReactor

BACKENDS = ["python"] + (["cpp"] if cpp_available() else [])
needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ core not built")


def no_feedback(**extra) -> ReactorParams:
    return ReactorParams(alpha_fuel=0.0, alpha_moderator=0.0, sigma_xenon=0.0, sigma_samarium=0.0, **extra)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("power", [1.0, 0.5])
def test_equilibrium_is_steady(backend, power):
    r = make_reactor(ReactorParams(), backend)
    r.reset(power, 0.25, 650.0)
    y0 = r.state
    r.advance(3600.0, 0.0, 0.0)
    assert np.allclose(r.state, y0, rtol=1e-9, atol=1e-9)
    source_supported = -r.params.generation_time * r.params.neutron_source / power
    assert r.reactivity() == pytest.approx(source_supported, rel=1e-6, abs=1e-15)


def test_jacobian_matches_finite_differences():
    p = ReactorParams()
    y = model.equilibrium_state(p, 0.7) * (1.0 + 0.02 * np.random.default_rng(1).standard_normal(13))
    J = model.jacobian(p, y, 0.3, 600.0, 0.08)
    fd = np.zeros_like(J)
    for j in range(13):
        h = 1e-6 * max(1.0, abs(y[j]))
        e = np.zeros(13)
        e[j] = h
        plus = model.derivatives(p, y + e, 0.3, 600.0, 0.08)
        minus = model.derivatives(p, y - e, 0.3, 600.0, 0.08)
        fd[:, j] = (plus - minus) / (2 * h)
    assert np.allclose(J, fd, rtol=1e-5, atol=1e-8)


@needs_cpp
def test_cpp_defaults_match_python():
    from norman import _core

    native, py = _core.Params(), ReactorParams()
    for name in ReactorParams.__dataclass_fields__:
        a, b = getattr(native, name), getattr(py, name)
        assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float)), name


@needs_cpp
def test_cpp_derivatives_match_python():
    p = ReactorParams()
    cpp, ref = CppReactor(p), PythonReactor(p)
    cpp.reset(1.0, 0.25, 650.0)
    ref.reset(1.0, 0.25, 650.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = model.equilibrium_state(p, rng.uniform(0.2, 1.0)) * (1 + 0.05 * rng.standard_normal(13))
        rod, boron = rng.uniform(0, 1), rng.uniform(0, 1500)
        assert np.allclose(cpp.derivatives(y, rod, boron), ref.derivatives(y, rod, boron), rtol=1e-12, atol=1e-14)


@needs_cpp
def test_cpp_transient_matches_scipy_radau():
    p = ReactorParams()
    cpp, ref = CppReactor(p), PythonReactor(p)
    for r in (cpp, ref):
        r.reset(1.0, 0.25, 650.0)
    rng = np.random.default_rng(3)
    for _ in range(30):
        rod_cmd, boron_cmd = rng.uniform(-0.3, 0.3), rng.uniform(-1, 1)
        cpp.advance(60.0, rod_cmd, boron_cmd)
        ref.advance(60.0, rod_cmd, boron_cmd)
    assert abs(cpp.power - ref.power) < 1e-4
    assert np.allclose(cpp.state, ref.state, rtol=1e-4, atol=1e-6)
    assert cpp.rod == pytest.approx(ref.rod) and cpp.boron == pytest.approx(ref.boron)


@pytest.mark.parametrize("backend", BACKENDS)
def test_prompt_jump(backend):
    p = no_feedback(substep=0.001)
    r = make_reactor(p, backend)
    r.reset(1.0, 0.25, 650.0)
    rho = 0.2 * p.beta
    r.set_state(r.state, r.rod, r.boron, r.bias + rho)
    r.advance(0.05, 0.0, 0.0)
    assert r.power == pytest.approx(p.beta / (p.beta - rho), rel=0.01)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rho_pcm", [20.0, 100.0])
def test_stable_period_matches_inhour_equation(backend, rho_pcm):
    p = no_feedback()
    r = make_reactor(p, backend)
    r.reset(1e-3, 0.25, 650.0)
    rho = rho_pcm * 1e-5
    r.set_state(r.state, r.rod, r.boron, r.bias + rho)
    expected_omega = model.inhour_root(p, rho)
    window = 2.0 / expected_omega
    r.advance(6.0 / expected_omega, 0.0, 0.0)
    p1 = r.power
    r.advance(window, 0.0, 0.0)
    measured_omega = math.log(r.power / p1) / window
    assert measured_omega == pytest.approx(expected_omega, rel=5e-3)


def analytic_xenon_after_shutdown(p: ReactorParams, t: np.ndarray) -> np.ndarray:
    li, lx, b = p.lambda_iodine, p.lambda_xenon, p.xenon_iodine_source
    return np.exp(-lx * t) + b / (lx - li) * (np.exp(-li * t) - np.exp(-lx * t))


@pytest.mark.parametrize("backend", BACKENDS)
def test_xenon_peak_after_shutdown(backend):
    p = ReactorParams()
    r = make_reactor(p, backend)
    r.reset(1.0, 0.25, 650.0)
    r.set_state(r.state, r.rod, r.boron, r.bias - 0.08)
    times, xenon = [0.0], [r.state[model.XENON]]
    for _ in range(24 * 12):
        r.advance(300.0, 0.0, 0.0)
        times.append(r.time)
        xenon.append(r.state[model.XENON])
    times, xenon = np.array(times), np.array(xenon)

    grid = np.linspace(0, 24 * 3600, 200001)
    exact = analytic_xenon_after_shutdown(p, grid)
    t_peak_exact = grid[np.argmax(exact)]
    t_peak = times[np.argmax(xenon)]

    infinite_flux_limit = math.log(p.lambda_iodine / p.lambda_xenon) / (p.lambda_iodine - p.lambda_xenon)
    assert 6.0 * 3600 < t_peak_exact < infinite_flux_limit
    assert abs(t_peak - t_peak_exact) < 0.25 * 3600
    assert xenon.max() == pytest.approx(exact.max(), rel=0.01)
    assert np.allclose(xenon, analytic_xenon_after_shutdown(p, times), rtol=0.01)


def test_rod_worth_curve():
    p = ReactorParams()
    z = np.linspace(0, 1, 101)
    w = np.array([model.rod_worth(p, v) for v in z])
    assert w[0] == 0.0
    assert w[-1] == pytest.approx(-p.rod_bank_worth)
    assert np.all(np.diff(w) <= 0)


def test_power_defect_matches_feedback_coefficients():
    p = ReactorParams()
    y100, y50 = model.equilibrium_state(p, 1.0), model.equilibrium_state(p, 0.5)
    defect = p.alpha_fuel * (y50[model.TF] - y100[model.TF]) + p.alpha_moderator * (y50[model.TC] - y100[model.TC])
    assert 300e-5 < defect < 1500e-5


def test_equilibrium_xenon_worth_is_textbook_magnitude():
    p = ReactorParams()
    assert 0.018 < p.xenon_worth_full_power < 0.030
    assert p.samarium_worth_full_power == pytest.approx(0.0113 / 2.43)
