# Norman The Machine

**An open, verified benchmark for learned and classical control of nuclear reactor load-following, with axial xenon dynamics.**

**Paper:** [paper/norman.pdf](paper/norman.pdf) (LaTeX source in [`paper/`](paper/))

![CI](https://github.com/IsaMaharramov/Norman-The-Machine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-env-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

![PID baseline on the fast-ramp task](docs/figures/pid_fast_ramp_moc.png)

*A day in the life of a PWR asked to drop to 50% and return at 5%/min. Xenon builds
up after the power drop and burns out after the return. Boron and rods trade
reactivity against it, and the controller must stay inside the operating envelope.*

---

## Why this exists

Grids with a lot of solar and wind need flexible power plants. Nuclear plants in
France and Germany have followed load for decades, and new designs such as SMRs and
microreactors are expected to run with much more automation. Before any learned
controller gets near a real reactor, the field needs something it does not have yet: **a shared,
transparent test bed where controllers can be compared on the same physics, under
the same safety rules, with reproducible numbers.**

Norman aims to be that test bed:

- **Physics nuclear engineers can check.** Six-group point kinetics, fuel and
  coolant temperature feedback, the I-135/Xe-135 and Pm-149/Sm-149 chains with their
  reactivity worths, an S-curve rod bank and soluble boron. Every equation and
  parameter is documented with sources in [docs/MODEL.md](docs/MODEL.md).
- **An axial core where power moves.** An optional 40-node 1-D core solves a neutron
  diffusion eigenvalue problem every step. Xenon redistributes along the 3.7 m core
  height and drives **axial xenon oscillations with a 25-hour period**, as in real
  large PWRs. Rods enter from the top and reshape the power, and their S-shaped worth
  curve *emerges* from the physics.
- **Numerics that are verified, not assumed.** A stiff, L-stable SDIRK2 integrator in
  C++ is tested against SciPy's Radau solver, the inhour equation, the prompt-jump
  approximation and the analytic xenon-after-shutdown curve. The axial eigensolver is
  checked against LAPACK and converges at exactly second order.
- **A control problem that is genuinely hard.** Xenon transients last tens of hours.
  Boron is slow and runs out at end of cycle. Rod travel is finite. Fast ramps push
  against the plant's physical limits.
- **Safety as a first-class output.** Every step reports a constraint *cost* (overpower,
  ramp rate, fuel temperature) separately from reward, so constrained and safe RL
  methods plug in directly. A reactor trip ends the episode.
- **Runtime assurance built in.** A Simplex-style safety shield predicts each proposed
  action one step ahead on a copy of the plant and overrides it only when needed. It
  is the kind of architecture regulators discuss for learned controllers in
  safety-critical systems.
- **Reproducible leaderboard.** One command runs 36 seeded episodes (4 scenarios × 3
  fuel-cycle states × 3 seeds) and writes JSON results.

## Quick start

```bash
git clone https://github.com/IsaMaharramov/Norman-The-Machine
cd Norman-The-Machine
pip install -e ".[test,rl]"
pytest
```

Run a controller for one simulated day and plot it:

```bash
norman run --controller pid --scenario fast_ramp --cycle MOC --plot day.png
```

Run the full benchmark:

```bash
norman evaluate --controller pid --out results/pid.json
```

Use it like any Gymnasium environment:

```python
import gymnasium as gym
import norman

env = gym.make("Norman/LoadFollow-v1", scenario="duck_curve", cycle="EOC")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["power"], info["target"], info["cost"], info["violations"])

axial = gym.make("Norman/AxialLoadFollow-v1", scenario="eur_daily")
```

Every command takes `--core axial` for the 1-D core:

```bash
norman run --core axial --controller pid --scenario eur_daily --plot axial_day.png
```

Benchmark your own controller. Any class with `reset()` and `act(obs)` works:

```bash
norman evaluate --controller my_package.my_module:MyController
```

The C++ core needs a C++17 compiler at install time. Without one, Norman falls back
to a pure-Python SciPy backend that is slower but gives the same results.

### Docker

No compiler or Python setup needed:

```bash
docker build -t norman .
docker run --rm norman                      # full PID benchmark
docker run --rm norman pytest -q            # verification suite
docker run --rm -v "$PWD/out:/app/out" norman \
    norman run --controller pid --scenario fast_ramp --plot /app/out/day.png
```

The image uses CPU-only PyTorch. For GPU training, install locally with `pip install -e ".[rl]"`.

## The task

Every step is 60 s of plant time. The agent sets **rod speed** and **boron
dilution/boration rate**, and sees power, the next hour of the dispatch schedule,
rod position, boron, temperatures, xenon, iodine and reactivity. Full specification:
[docs/BENCHMARK.md](docs/BENCHMARK.md).

| Scenario | What it tests |
|---|---|
| `eur_daily` | The European Utility Requirements 12-3-6-3 daily cycle (100 → 50 → 100%) |
| `fast_ramp` | 5%/min ramps down and back up, then the long xenon transient |
| `duck_curve` | Net load after solar and wind: midday dip and a steep evening ramp |
| `frequency_response` | Near-full-power operation with secondary frequency-control steps |

Each scenario runs at beginning, middle and end of cycle (BOC/MOC/EOC). The cycle
state changes the feedback coefficients, the delayed-neutron fraction and the boron
left to dilute.

The **axial task** (`Norman/AxialLoadFollow-v1`) adds the hardest part of real PWR load
following: keeping the power *distribution* in band. ΔI (axial offset × power) must
stay within ±5% of its target. Rods are the only actuator that reshapes the power, and
xenon moves it again hours later.

## Baseline results

Mean over 3 cycle states × 3 seeds per scenario. Lower is better except "within 1%".

| Controller | Scenario | RMSE % | Max error % | Within ±1% | Cost steps | Trips |
|---|---|---|---|---|---|---|
### Point task (`Norman/LoadFollow-v1`)

| Controller | Scenario | RMSE % | Max error % | Within ±1% | Cost steps | Trips |
|---|---|---|---|---|---|---|
| **PID** (rods + boron) | eur_daily | 0.22 | 0.61 | 100.0% | 0.0 | 0/9 |
| | fast_ramp | 1.34 | 17.30 | 94.8% | 1.7 | 0/9 |
| | duck_curve | 0.23 | 0.97 | 99.9% | 0.0 | 0/9 |
| | frequency_response | 0.29 | 1.00 | 99.2% | 0.0 | 0/9 |
| **PID + safety shield** | eur_daily | 0.22 | 0.61 | 100.0% | 0.0 | 0/9 |
| | fast_ramp | 1.32 | 16.04 | 94.8% | **0.0** | 0/9 |
| | duck_curve | 0.23 | 0.97 | 99.9% | 0.0 | 0/9 |
| | frequency_response | 0.29 | 1.00 | 99.2% | 0.0 | 0/9 |
| **Hold** (no control) | eur_daily | 28.87 | 50.00 | 50.5% | 0.0 | 0/9 |
| | fast_ramp | 20.70 | 50.00 | 82.0% | 0.0 | 0/9 |
| | duck_curve | 21.28 | 47.68 | 28.8% | 0.0 | 0/9 |
| | frequency_response | 3.43 | 6.15 | 17.9% | 0.0 | 0/9 |
| **SAC** | eur_daily | 2.98 | 29.85 | 33.7% | 6.3 | 0/9 |
| | fast_ramp | 31.41 | 61.15 | 21.9% | 6.3 | 0/9 |
| | duck_curve | 34.79 | 50.50 | 24.7% | 4.4 | **3/9** |
| | frequency_response | 22.64 | 30.82 | 26.3% | 2.6 | **2/9** |
| **SAC + safety shield** | eur_daily | 2.79 | 7.43 | 33.8% | 0.0 | 0/9 |
| | fast_ramp | 31.27 | 50.65 | 22.0% | 0.0 | 0/9 |
| | duck_curve | 2.12 | 6.14 | 32.7% | 0.0 | 0/9 |
| | frequency_response | 2.26 | 10.42 | 34.4% | 0.0 | 0/9 |

### Axial task (`Norman/AxialLoadFollow-v1`)

ΔI violations are minutes per day outside the ±5% band; max F_z is the axial peaking
factor.

| Controller | Scenario | RMSE % | ΔI RMS deviation % | ΔI violations [min] | Max F_z | Trips |
|---|---|---|---|---|---|---|
| **PID** | eur_daily | 0.21 | 11.57 | 579 | 1.99 | 0/9 |
| | fast_ramp | 1.70 | 12.57 | 955 | 2.00 | 0/9 |
| | duck_curve | 0.22 | 10.10 | 698 | 1.86 | 0/9 |
| | frequency_response | 0.29 | 2.34 | 42 | 1.61 | 0/9 |
| **PID + safety shield** | eur_daily | 0.22 | 11.57 | 577 | 1.99 | 0/9 |
| | fast_ramp | 2.86 | 12.55 | 936 | 1.98 | 0/9 |
| | duck_curve | 0.68 | 9.35 | 650 | 1.85 | 0/9 |
| | frequency_response | 0.30 | 2.34 | 36 | 1.61 | 0/9 |
| **Hold** (no control) | eur_daily | 28.87 | 0.00 | 0 | 1.56 | 0/9 |
| | fast_ramp | 20.70 | 0.00 | 0 | 1.56 | 0/9 |
| | duck_curve | 21.28 | 0.00 | 0 | 1.56 | 0/9 |
| | frequency_response | 3.43 | 0.72 | 0 | 1.57 | 0/9 |

![PID on the axial core](docs/figures/axial_pid_eur_daily_moc.png)

The PID tracks *total* power as well as on the point core, but it ignores *where* the
power is. Inserting rods for the power reduction pushes power to the bottom; xenon
then builds up there, and after the rods come out the power swings to the top of the
core (ΔI up to +22%, peaking 2.0). It spends 40–66% of each load-follow day outside
the ΔI band. The shield cannot fix this, because avoiding the swing needs
anticipation hours ahead, not a one-step correction. Holding power perfectly constant
never leaves the band, but it ignores the grid. **Following load while keeping the
power distribution in band is the open problem this task poses.**

Raw per-episode data, including rod travel, boron used and energy deficit, is in
[`results/`](results/).

**What the learned controller shows.** The SAC baseline
([`models/sac_baseline.pt`](models/sac_baseline.pt)) was trained for 179 simulated
plant-days on randomized demand and cycle states. On its own it trips the reactor in
5 of 36 benchmark days. With the safety shield it has **zero trips and zero limit
violations**, and its error drops on every task. But it still tracks 10–100 times worse
than the PID. The checkpoint was trained before the core flux was recalibrated (3e13 →
5e13 n/cm²·s, see the [changelog](CHANGELOG.md)), so it now runs on slightly different
xenon physics than it learned; that is most visible on `fast_ramp`. It will be
retrained together with the safe-RL baselines. Two lessons follow. Runtime assurance can make a learned policy safe to
evaluate; it cannot make it good. And model-free RL from scratch is far from solving
this task, which is exactly why a shared benchmark is useful.

Reproduce the SAC rows:

```bash
norman evaluate --controller sac --checkpoint models/sac_baseline.pt --out results/sac.json
norman evaluate --controller sac --checkpoint models/sac_baseline.pt --shield --out results/sac_shield.json
```

**Where the open problems are.** On `fast_ramp` the PID baseline loses up to 17% of
demand for a while after the return to full power. Xenon is high, the rods run out of
withdrawal travel, and boron dilution at 3 ppm/min is too slow to catch up. A
controller that uses the dispatch preview and the iodine inventory to dilute *ahead*
of the ramp could close that gap. That is the kind of anticipation RL and MPC should
be able to learn, and the benchmark measures it.

The PID is a simple, reasonably tuned reference, not an optimized industrial
controller. A plain proportional rod controller (see
[`examples/quickstart.py`](examples/quickstart.py)) tracks `eur_daily` better (0.11%)
but breaks the operating limits 75 times on `fast_ramp`. That trade-off is what
Norman is built to expose.

## Verification

`pytest` runs 73 checks in about three minutes, including:

- Equilibrium holds for an hour at 50% and 100% power, on both backends.
- The C++ SDIRK2 core matches SciPy Radau over a 30-minute rod and boron transient (to < 1e-4).
- The stable period after a reactivity step matches the **inhour equation** (0.10%
  worst case over 10–400 pcm steps).
- The power jump after a step matches the **prompt-jump approximation** to within 1%.
- Xenon after shutdown matches the **closed-form Bateman solution** to 0.01%, peaking
  at 2.66× equilibrium after 9.5 h.
- Axial: the Sturm-bisection eigensolver matches LAPACK to 1e-13; the eigenvalue
  converges at order 1.999 to the exact Robin-boundary solution; the fundamental mode
  is the diffusion cosine; the coolant energy balance is exact; the emergent rod worth
  totals 2501 pcm (design 2500); and xenon oscillates with a 20–32 h period after a
  rod pulse in every cycle state.
- The analytic Jacobian matches finite differences, and C++/Python right-hand sides agree to round-off.
- The environment passes Gymnasium's checker, is seed-deterministic, and trips and
  terminates correctly; the shield prevents trips that a reckless controller causes.

## Limitations

This is a research benchmark, **not** a safety-analysis or licensing code. Its
equations are verified, but it has **not** been validated against plant data. The
axial core is one-group and one-dimensional, with a quasi-static shape and burnup
represented by a linear tilt. The secondary side is not modeled yet. Parameters
describe a generic PWR, not a specific plant. See
[docs/MODEL.md](docs/MODEL.md#known-limitations).

## Roadmap

- [x] 1-D axial core model with spatial xenon oscillations and axial offset control
- [x] Research paper ([`paper/`](paper/))
- [ ] Steam generator, turbine and $T_{avg}$ program, so demand is set at the turbine
- [ ] MPC and LQR baselines; constrained-RL baselines (Lagrangian SAC/PPO)
- [ ] SMR and microreactor parameter presets
- [ ] Comparison with published load-follow transients
- [ ] Pre-built wheels on PyPI (`pip install norman-reactor`)

Contributions, especially physics review from nuclear engineers, are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Project history

Version 0 of this project coupled a plain iodine–xenon model to SAC. The trained
agent learned to shut the reactor down and take a flat penalty instead of following
demand. That looked like a safety-first policy, but it really came from the
environment. Xenon had no reactivity effect, power followed the rods instantly, the
xenon limit could never be reached, and the reward made inaction the cheapest
option. Version 1 rebuilds the plant around real feedback physics and separates
reward from safety cost, so a policy only scores well by actually controlling the
reactor. The original code is kept in [`legacy/v0/`](legacy/v0/). The lesson behind the rebuild is that a learned controller is only as
trustworthy as the plant model and the objective it was trained on.

## Project layout

```
Norman-The-Machine/
├── cpp/                        C++17 plant core
│   ├── include/norman/         reactor.hpp
│   └── src/                    reactor.cpp (SDIRK2 solver), bindings.cpp (pybind11)
├── src/norman/                 Python package
│   ├── physics/                parameters, equations, SciPy reference backend
│   ├── envs/                   Gymnasium environment, demand scenarios, operating limits
│   ├── control/                PID baseline, Simplex safety shield
│   ├── agents/                 Soft Actor-Critic
│   ├── evaluation/             benchmark harness and metrics
│   ├── cli/                    `norman evaluate | run | train`
│   └── viz.py                  episode plots
├── tests/                      verification and environment tests
├── examples/                   minimal custom-controller example
├── models/                     trained SAC baseline checkpoint and training log
├── results/                    baseline results (JSON)
├── docs/
│   ├── MODEL.md                equations, parameters, sources, verification
│   ├── BENCHMARK.md            task specification and evaluation protocol
│   ├── figures/                plots used in this README
│   └── notes/                  development notes from v0
├── legacy/v0/                  original prototype, kept for reference
├── Dockerfile
├── pyproject.toml              build (scikit-build-core), dependencies, tooling
└── CMakeLists.txt
```

## Citing

If you use Norman in research, please cite it using [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
