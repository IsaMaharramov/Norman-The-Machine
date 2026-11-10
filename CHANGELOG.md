# Changelog

## 1.0.0 — 2026-11-11

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
- Verification test suite, model and benchmark specifications, Docker image, CI.

### Moved
- The v0 simulator (`core/`, `envs/`, `agent/`, `scripts/`, `train.py`, `test.py`) and
  its checkpoint now live in `legacy/v0/`. They are not compatible with the new model.

## 0.x

Original prototype: iodine–xenon equations coupled to SAC.
