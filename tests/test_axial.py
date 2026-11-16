import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from norman.control.pid import PIDController
from norman.control.shield import SafetyShield
from norman.envs.load_follow import AXIAL_OBS, LoadFollowEnv
from norman.evaluation import run_episode
from norman.physics.axial import AxialParams, CppAxialCore, PythonAxialCore, leakage_matrix, make_axial_core
from norman.physics.backend import cpp_available
from norman.physics.params import CYCLE_INITIAL_BORON, cycle_params
from norman.physics.plant import AxialPlant

BACKENDS = ["python"] + (["cpp"] if cpp_available() else [])
needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ core not built")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rho", [0.0, -0.002, 0.004])
def test_uniform_local_reactivity_is_global_reactivity(backend, rho):
    core = make_axial_core(backend=backend)
    lam, _ = core.solve_eigen(np.full(core.nodes, rho))
    assert 1.0 - lam == pytest.approx(rho, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fundamental_mode_is_the_diffusion_cosine(backend):
    a = AxialParams()
    core = make_axial_core(axial=a, backend=backend)
    _, shape = core.solve_eigen(np.zeros(a.nodes))
    z = (np.arange(a.nodes) + 0.5) * a.height / a.nodes
    exact = np.cos(math.pi * (z - a.height / 2) / (a.height + 2 * a.extrapolation))
    exact /= exact.mean()
    assert np.max(np.abs(shape - exact)) < 2e-3


def robin_eigenvalue(a: AxialParams) -> float:
    from scipy.optimize import brentq

    b = brentq(lambda x: x * math.tan(x * a.height / 2) - 1.0 / a.extrapolation, 1e-9, math.pi / a.height - 1e-12)
    return 1.0 + a.migration_area * b**2


def test_leakage_eigenvalue_converges_at_second_order():
    def error(nodes):
        a = AxialParams(nodes=nodes)
        return abs(PythonAxialCore(axial=a).k_infinity_reference - robin_eigenvalue(a))

    e40, e80, e160 = error(40), error(80), error(160)
    assert e40 < 2e-6
    assert e40 / e80 == pytest.approx(4.0, rel=0.1)
    assert e80 / e160 == pytest.approx(4.0, rel=0.1)


@needs_cpp
def test_sturm_bisection_matches_lapack():
    from norman import _core

    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.uniform(1.0, 3.0, 40)
        e = rng.uniform(-1.0, 0.0, 39)
        from scipy.linalg import eigh_tridiagonal

        exact = eigh_tridiagonal(d, e, eigvals_only=True, select="i", select_range=(0, 0))[0]
        assert _core.smallest_eigenvalue(list(d), list(e)) == pytest.approx(exact, abs=1e-13)
    diag, off = leakage_matrix(AxialParams())
    assert _core.smallest_eigenvalue(list(diag), list(off)) == pytest.approx(
        PythonAxialCore().k_infinity_reference, abs=1e-13)


@needs_cpp
def test_cpp_axial_core_matches_python_reference():
    p = cycle_params("MOC")
    cpp, ref = CppAxialCore(p), PythonAxialCore(p)
    for core in (cpp, ref):
        core.initialize(0.8, 0.25, 650.0)
    assert np.allclose(cpp.shape, ref.shape, atol=1e-10)
    assert cpp.tilt == pytest.approx(ref.tilt, rel=1e-9)
    rho = np.random.default_rng(1).normal(0.0, 0.003, cpp.nodes)
    lc, sc = cpp.solve_eigen(rho)
    lr, sr = ref.solve_eigen(rho)
    assert lc == pytest.approx(lr, abs=1e-13)
    assert np.allclose(sc, sr, atol=1e-11)
    for _ in range(60):
        for core in (cpp, ref):
            core.advance_poisons(60.0, 0.6)
            core.update(0.6, 0.45, 650.0)
    assert np.allclose(cpp.xenon, ref.xenon, rtol=1e-5)
    assert np.allclose(cpp.shape, ref.shape, atol=1e-5)
    assert cpp.reactivity == pytest.approx(ref.reactivity, abs=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rod_worth_curve_emerges_as_an_s_curve(backend):
    p = cycle_params("MOC")
    core = make_axial_core(p, backend=backend)
    core.initialize(1.0, 0.0, 650.0)
    worth = []
    for z in np.linspace(0.0, 1.0, 21):
        probe = core.copy()
        worth.append(probe.update(1.0, z, 650.0))
    worth = (np.array(worth) - worth[0]) / p.rod_bank_worth
    differential = -np.diff(worth)
    assert worth[-1] == pytest.approx(-1.0, abs=0.02)
    assert np.all(np.diff(worth) <= 1e-12)
    assert differential[0] < 0.25 * differential.max()
    assert differential[-1] < 0.25 * differential.max()
    assert 4 <= int(np.argmax(differential)) <= 16


@pytest.mark.parametrize("backend", BACKENDS)
def test_initialization_meets_design_offset_and_criticality(backend):
    core = make_axial_core(cycle_params("MOC"), backend=backend)
    core.initialize(1.0, 0.25, 650.0)
    assert core.axial_offset == pytest.approx(AxialParams().design_axial_offset, abs=1e-9)
    assert abs(core.reactivity) < 1e-10
    assert core.shape.mean() == pytest.approx(1.0)
    assert 1.2 < core.peaking < 1.8


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("power", [1.0, 0.6])
def test_coolant_energy_balance(backend, power):
    p = cycle_params("MOC")
    core = make_axial_core(p, backend=backend)
    core.initialize(power, 0.25, 650.0)
    t_in = p.t_inlet
    for tc in core.coolant_temperature:
        t_in = 2.0 * tc - t_in
    assert t_in == pytest.approx(p.t_inlet + power * p.rated_power / p.coolant_flow_capacity, abs=1e-9)
    assert t_in - p.t_inlet == pytest.approx(2.0 * (p.t_coolant_nominal - p.t_inlet) * power)


@pytest.mark.parametrize("backend", BACKENDS)
def test_axial_plant_equilibrium_is_steady(backend):
    plant = AxialPlant(cycle_params("MOC"), backend=backend)
    plant.reset(0.8, 0.25, 650.0)
    ao = plant.axial_offset
    for _ in range(30):
        plant.advance(60.0, 0.0, 0.0)
    assert plant.power == pytest.approx(0.8, abs=1e-7)
    assert plant.axial_offset == pytest.approx(ao, abs=1e-7)


@pytest.mark.parametrize("interval", [60.0, 5.0, 1.0])
def test_coupling_is_stable_at_any_interval(interval):
    plant = AxialPlant(cycle_params("MOC"), coupling_interval=interval)
    plant.reset(1.0, 0.25, 650.0)
    for _ in range(5):
        plant.advance(60.0, 0.0, 0.0)
        assert plant.peak_power < 1.0 + 1e-7
    assert plant.power == pytest.approx(1.0, abs=1e-7)


@needs_cpp
def test_coupling_converges_with_the_interval():
    rng = np.random.default_rng(4)
    commands = [(rng.uniform(-0.6, 0.6), rng.uniform(-1, 1)) for _ in range(40)]

    def trajectory(interval):
        plant = AxialPlant(cycle_params("MOC"), coupling_interval=interval)
        plant.reset(1.0, 0.25, 650.0)
        out = []
        for rod, boron in commands:
            plant.advance(60.0, rod, boron)
            out.append((plant.power, plant.axial_offset))
        return np.array(out)

    fine, mid, coarse = trajectory(1.25), trajectory(2.5), trajectory(5.0)
    e_mid, e_coarse = np.max(np.abs(mid - fine)), np.max(np.abs(coarse - fine))
    assert e_coarse < 0.02
    assert e_coarse > 1.5 * e_mid


@needs_cpp
def test_rod_worth_curve_matches_python_reference():
    p = cycle_params("MOC")
    cpp, ref = CppAxialCore(p), PythonAxialCore(p)
    for core in (cpp, ref):
        core.initialize(0.7, 0.3, 650.0)
    assert np.allclose(cpp.rod_worth_curve(21, 650.0), ref.rod_worth_curve(21, 650.0), atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_model_uses_the_installed_rod_worth_table(backend):
    from norman.physics.backend import make_reactor

    r = make_reactor(cycle_params("MOC"), backend)
    r.reset(1.0, 0.3, 650.0)
    table = np.linspace(0.0, -0.02, 11)
    r.set_rod_worth_table(table)
    assert r.reactivity_components()["rods"] == pytest.approx(-0.006)


@needs_cpp
@pytest.mark.parametrize("cycle", ["BOC", "MOC", "EOC"])
def test_axial_xenon_oscillation_period(cycle):
    plant = AxialPlant(cycle_params(cycle).replace(boron_rate_max=1.0))
    plant.reset(1.0, 0.25, CYCLE_INITIAL_BORON[cycle] + 200.0)

    def hold_power():
        return float(np.clip(2.0 * (1.0 - plant.power), -1.0, 1.0))

    plant.advance(60.0, -0.45, 0.0)
    for _ in range(120):
        plant.advance(60.0, 0.0, hold_power())
    plant.advance(60.0, 0.45, 0.0)
    offsets = []
    for _ in range(72 * 60):
        plant.advance(60.0, 0.0, hold_power())
        offsets.append(plant.axial_offset)
    offsets = np.array(offsets)
    centered = offsets - offsets[-600:].mean()
    crossings = np.where(np.diff(np.sign(centered)))[0]
    period_h = 2.0 * np.median(np.diff(crossings)) / 60.0
    assert 20.0 < period_h < 32.0
    first, later = np.ptp(offsets[:1500]), np.ptp(offsets[-1500:])
    assert later < first


def test_axial_env_passes_gymnasium_checks():
    env = LoadFollowEnv(core="axial", hours=1.0)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (18,)
    assert obs[AXIAL_OBS["delta_i_error_band"]] == pytest.approx(0.0, abs=1e-6)
    _, _, _, _, info = env.step(np.zeros(2))
    assert "axial_offset" in info["violations"]


def test_rod_insertion_pushes_power_to_the_bottom():
    env = LoadFollowEnv(core="axial", hours=1.0)
    env.reset(seed=0)
    before = env.reactor.axial_offset
    env.step(np.array([-0.4, 0.0]))
    assert env.reactor.rod == pytest.approx(0.25 + 0.4 * 60 / 180)
    assert env.reactor.axial_offset < before - 0.02


def test_shield_works_with_the_axial_plant():
    env = SafetyShield(LoadFollowEnv(core="axial", hours=2.0))
    _, m = run_episode(env, PIDController(), seed=0)
    assert not m["trip"]
    assert "delta_i_rms_pct" in m
