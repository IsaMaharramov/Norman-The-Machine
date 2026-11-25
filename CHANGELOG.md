# Changelog

## Unreleased (2.0 in progress)

### Added
- 1-D axial core (`core="axial"`, `Norman/AxialLoadFollow-v1`): 40 nodes with local
  temperatures and I/Xe/Pm/Sm, one-group diffusion eigenvalue shape solved by
  Sturm-sequence bisection, quasi-static coupling to point kinetics, rods entering from
  the top, and upward coolant heating. It reproduces axial xenon oscillations with a
  25 h period and an emergent, skewed S-shaped rod worth curve.
- Independent Python reference for the axial core (LAPACK eigensolver, exact matrix
  exponential for the poisons).
- Quasi-static coupling every 5 s inside each 60 s control step. Local temperatures
  carry the point model's thermal lag, and a live rod worth table is computed from the
  current flux shape. A convergence study shows first-order accuracy, about 1% of
  power at 5 s.
- Optional tabulated rod worth in the point model (`rod_worth_table`).
- Steam generators and turbine (`secondary=True`, `Norman/PlantLoadFollow-v1`): hot
  leg, steam-generator primary and secondary nodes, cold leg, and a turbine whose
  output is limited by steam pressure. Steam pressure comes from the IAPWS-IF97
  saturation line and slides from 7.66 MPa at no load to 6.91 MPa at full load.
- T_avg program, steam-pressure limits, plant observations and metrics, and a
  Westinghouse-style T_avg rod controller (`--controller tavg`).
- `--plant` option for `norman evaluate`, `run` and `train`; plant episode plots.
- Axial-offset (ΔI band) operating limit, four axial observations, and axial metrics
  (ΔI deviation, peaking, xenon offset swing).
- `--core axial` option for `norman evaluate`, `run` and `train`; axial episode plots
  with power and xenon heat maps.
- Research paper in `paper/` describing the model, its verification and the baselines.

### Fixed
- `norman evaluate` wrote no results file on Windows consoles when the table contained
  non-ASCII characters. Results are now written first and the table header is ASCII.

### Changed
- Core-average thermal flux raised from 3e13 to 5e13 n/cm²/s. Equilibrium xenon worth
  rises from 2160 to 2360 pcm, closer to real PWRs, and the axial oscillation period
  reaches the 25 h characteristic of large PWRs. All baseline results were regenerated.

## 1.0.0 (2026-11-11)

Complete rebuild as a benchmark.

### Added
- C++17 plant core: six-group point kinetics, fuel and coolant temperature feedback,
  I-135/Xe-135 and Pm-149/Sm-149 with reactivity worth, S-curve rod bank, soluble boron,
  L-stable SDIRK2 integrator with analytic Jacobian.
- Pure-Python SciPy Radau backend used as the independent reference.
- `Norman/LoadFollow-v1` Gymnasium environment with four demand scenarios, BOC/MOC/EOC
  presets, domain randomization, per-step safety cost and reactor trip.
- PID baseline, Simplex safety shield, Soft Actor-Critic agent.
- `norman` command with `evaluate`, `run` and `train` subcommands.
- Trained SAC baseline checkpoint (`models/sac_baseline.pt`) and its benchmark results.
- Verification test suite, model and benchmark specifications, Docker image, CI.

### Moved
- The v0 simulator (`core/`, `envs/`, `agent/`, `scripts/`, `train.py`, `test.py`) and
  its checkpoint now live in `legacy/v0/`. They are not compatible with the new model.

## 0.x

Original prototype: iodine–xenon equations coupled to SAC.
