# Contributing

Contributions from both nuclear engineers and control/ML researchers are welcome.
The most valuable ones are:

1. **Physics review.** If a parameter, equation or simplification is wrong or
   misleading for a PWR, open an issue with a reference. Model credibility comes first.
2. **Model extensions** from the roadmap in the README: 1-D axial core with xenon
   oscillations, steam generator and turbine, multiple rod banks, SMR presets.
3. **New controllers.** MPC, LQR or gain-scheduled controllers, constrained RL, and
   imitation of the PID with safe improvement.
4. **Benchmark results.** Run the protocol in [docs/BENCHMARK.md](docs/BENCHMARK.md)
   and submit `results/<name>.json`.

## Development setup

```bash
git clone https://github.com/IsaMaharramov/Norman-The-Machine
cd Norman-The-Machine
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,rl]"
pytest
```

Building the C++ core needs a C++17 compiler: MSVC Build Tools on Windows, gcc or
clang elsewhere. Without one, use the pure-Python backend: pass `backend="python"`.

## Rules for physics changes

- Change the equations in **both** `src/norman/physics/model.py` and `cpp/src/reactor.cpp`.
  The parity tests fail if they disagree.
- Add a verification test against an analytic or published result.
- Update the parameter table and equations in `docs/MODEL.md`.
- If a change moves the baseline numbers, regenerate `results/` and the README table
  in the same pull request.
