import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from norman.control.pid import TavgController
from norman.envs.load_follow import LoadFollowEnv
from norman.evaluation import run_episode
from norman.physics import model
from norman.physics.backend import CppReactor, cpp_available, make_reactor
from norman.physics.params import ReactorParams, cycle_params
from norman.physics.reference import PythonReactor
from norman.physics.water import saturation_pressure

BACKENDS = ["python"] + (["cpp"] if cpp_available() else [])
needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ core not built")

IF97_TABLE = [(300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)]


@pytest.mark.parametrize("kelvin,pressure", IF97_TABLE)
def test_saturation_pressure_matches_if97_verification_table(kelvin, pressure):
    assert saturation_pressure(kelvin - 273.15) == pytest.approx(pressure, rel=1e-8)


@needs_cpp
@pytest.mark.parametrize("kelvin,pressure", IF97_TABLE)
def test_cpp_saturation_pressure_matches_if97(kelvin, pressure):
    from norman import _core

    assert _core.saturation_pressure(kelvin - 273.15) == pytest.approx(pressure, rel=1e-8)


def test_steam_pressure_slides_down_with_load():
    p = ReactorParams(secondary=True)
    pressures = [saturation_pressure(model.equilibrium_state(p, load)[model.TS]) for load in (0.01, 0.5, 1.0)]
    assert pressures[0] > pressures[1] > pressures[2]
    assert pressures[0] == pytest.approx(7.66, abs=0.02)
    assert pressures[2] == pytest.approx(6.91, abs=0.02)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("power", [1.0, 0.5])
def test_plant_equilibrium_is_steady_and_on_program(backend, power):
    p = cycle_params("MOC").replace(secondary=True)
    r = make_reactor(p, backend)
    r.reset(power, 0.25, 650.0)
    y0 = r.state
    r.advance(1800.0, 0.0, 0.0, power)
    assert np.allclose(r.state, y0, rtol=1e-9, atol=1e-8)
    assert r.state[model.TC] == pytest.approx(p.tavg_reference(power))
    assert r.steam_power == pytest.approx(power)


def test_energy_balance_at_steady_state():
    p = ReactorParams(secondary=True)
    for load in (0.3, 0.7, 1.0):
        y = model.equilibrium_state(p, load)
        core = 2.0 * p.coolant_flow_capacity * (y[model.TC] - y[model.TCL])
        generator = p.sg_conductance * (y[model.TP] - y[model.TS])
        assert core == pytest.approx(p.rated_power * load)
        assert generator == pytest.approx(p.rated_power * load)


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_regulation_matches_the_feedback_balance(backend):
    p = cycle_params("MOC").replace(secondary=True, sigma_xenon=0.0, sigma_samarium=0.0)
    r = make_reactor(p, backend)
    r.reset(0.8, 0.25, 650.0)
    t_before = r.state[model.TC]
    r.advance(60.0, 0.0, 0.0, 0.85)
    for _ in range(29):
        r.advance(60.0, 0.0, 0.0)
    gap = p.t_fuel_nominal - p.t_coolant_nominal
    expected_shift = -p.alpha_fuel * gap * 0.05 / (p.alpha_fuel + p.alpha_moderator)
    assert r.power == pytest.approx(0.85, abs=1e-6)
    assert r.steam_power == pytest.approx(0.85, abs=1e-9)
    assert r.state[model.TC] - t_before == pytest.approx(expected_shift, abs=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_load_step_without_control_starves_the_turbine(backend):
    p = cycle_params("MOC").replace(secondary=True, sigma_xenon=0.0, sigma_samarium=0.0)
    r = make_reactor(p, backend)
    r.reset(0.9, 0.25, 650.0)
    r.advance(60.0, 0.0, 0.0, 1.0)
    for _ in range(29):
        r.advance(60.0, 0.0, 0.0)
    available = p.turbine_valve_max * saturation_pressure(r.state[model.TS]) / p.steam_pressure_full
    assert r.steam_power == pytest.approx(available)
    assert 0.97 < r.steam_power < 0.995
    assert r.power == pytest.approx(r.steam_power, abs=1e-6)


@needs_cpp
def test_cpp_plant_transient_matches_scipy_radau():
    p = cycle_params("MOC").replace(secondary=True)
    cpp, ref = CppReactor(p), PythonReactor(p)
    for r in (cpp, ref):
        r.reset(1.0, 0.25, 650.0)
    rng = np.random.default_rng(5)
    for _ in range(30):
        rod_cmd, boron_cmd, load = rng.uniform(-0.3, 0.3), rng.uniform(-1, 1), rng.uniform(0.6, 1.0)
        cpp.advance(60.0, rod_cmd, boron_cmd, load)
        ref.advance(60.0, rod_cmd, boron_cmd, load)
    assert np.allclose(cpp.state, ref.state, rtol=1e-4, atol=1e-6)
    assert cpp.steam_power == pytest.approx(ref.steam_power, rel=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_turbine_output_is_limited_by_steam_pressure(backend):
    p = cycle_params("MOC").replace(secondary=True)
    r = make_reactor(p, backend)
    r.reset(1.0, 0.25, 650.0)
    r.advance(60.0, 0.0, 0.0, 1.2)
    available = p.turbine_valve_max * saturation_pressure(r.state[model.TS]) / p.steam_pressure_full
    assert r.steam_power == pytest.approx(available)
    assert r.steam_power < 1.2


def test_plant_env_passes_gymnasium_checks():
    env = LoadFollowEnv(secondary=True, hours=1.0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,)
    _, _, _, _, info = env.step(np.zeros(2))
    assert "steam_pressure" in info["violations"]
    assert info["steam_pressure"] == pytest.approx(6.91, abs=0.05)


def test_axial_plant_env_runs():
    env = LoadFollowEnv(secondary=True, core="axial", hours=0.5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (22,)
    for _ in range(env.steps):
        obs, _, terminated, _, info = env.step(np.zeros(2))
        assert not terminated
    assert abs(info["tavg_error"]) < 1.0


def test_load_rejection_without_control_lifts_steam_pressure():
    env = LoadFollowEnv(scenario="fast_ramp", secondary=True, hours=3.0)
    obs, _ = env.reset(seed=0)
    pressures = []
    for _ in range(env.steps):
        obs, _, _, _, info = env.step(np.zeros(2))
        pressures.append(info["steam_pressure"])
    assert max(pressures) > 8.0


def test_tavg_controller_holds_the_program():
    env = LoadFollowEnv(scenario="eur_daily", cycle="MOC", secondary=True)
    _, m = run_episode(env, TavgController(), seed=0)
    assert not m["trip"]
    assert m["cost_steps"] == 0
    assert m["tavg_max_k"] < 1.5
    assert m["rmse_pct"] < 0.5
