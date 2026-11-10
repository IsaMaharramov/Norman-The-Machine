from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from pathlib import Path

from norman import __version__
from norman.control.pid import HoldController, PIDController
from norman.control.shield import SafetyShield
from norman.envs.load_follow import LoadFollowEnv
from norman.envs.scenarios import BENCHMARK_SCENARIOS, SCENARIOS
from norman.metrics import aggregate, episode_metrics

CYCLES = ("BOC", "MOC", "EOC")


def make_controller(name: str, checkpoint: str | None = None, boron_control: bool = True):
    if name == "pid":
        return PIDController(boron_control=boron_control)
    if name == "hold":
        return HoldController()
    if name == "sac":
        if not checkpoint:
            raise SystemExit("--checkpoint is required for the sac controller")
        from norman.agents.sac import SACAgent

        return SACAgent.load(checkpoint)
    if ":" in name:
        module, attr = name.split(":", 1)
        return getattr(importlib.import_module(module), attr)()
    raise SystemExit(f"unknown controller '{name}'")


def run_episode(env, controller, seed: int):
    obs, _ = env.reset(seed=seed)
    controller.reset()
    done = False
    info = {}
    while not done:
        action = controller.act(obs)[: env.action_space.shape[0]]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    plant = env.unwrapped
    metrics = episode_metrics(
        plant.history, plant.steps, plant.minutes_per_step,
        rated_power_mw=plant.reactor.params.rated_power / 1e6,
        interventions=info.get("shield_interventions", 0),
    )
    return plant.history, metrics


def evaluate(controller_name: str, scenarios, cycles, seeds, shield: bool = False,
             checkpoint: str | None = None, backend: str = "auto", hours: float = 24.0,
             boron_control: bool = True, verbose: bool = True) -> list[dict]:
    rows = []
    controller = make_controller(controller_name, checkpoint, boron_control)
    for scenario in scenarios:
        for cycle in cycles:
            env = LoadFollowEnv(scenario=scenario, cycle=cycle, backend=backend, hours=hours,
                                boron_control=boron_control)
            if shield:
                env = SafetyShield(env)
            for seed in seeds:
                start = time.perf_counter()
                _, m = run_episode(env, controller, seed)
                row = {"scenario": scenario, "cycle": cycle, "seed": seed, **m,
                       "wall_s": time.perf_counter() - start}
                rows.append(row)
                if verbose:
                    print(f"  {scenario:<19} {cycle} seed={seed}  rmse={m['rmse_pct']:.2f}%  "
                          f"max={m['max_abs_error_pct']:.2f}%  cost={m['cost_steps']}  trip={m['trip']}")
    return rows


def markdown_table(name: str, rows: list[dict]) -> str:
    lines = [
        f"| {name} | RMSE % | max err % | within 1% | energy deficit MWh | rod strokes | "
        "dilution ppm | boration ppm | cost steps | trips |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        a = aggregate([r for r in rows if r["scenario"] == scenario])
        lines.append(
            f"| {scenario} | {a['rmse_pct']:.2f} | {a['max_abs_error_pct']:.2f} | "
            f"{a['time_within_1pct'] * 100:.1f}% | {a['energy_deficit_mwh']:.0f} | "
            f"{a['rod_travel_strokes']:.2f} | {a['boron_dilution_ppm']:.1f} | {a['boration_ppm']:.1f} | "
            f"{a['cost_steps']:.1f} | {a['trips']}/{a['episodes']} |"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the Norman load-following benchmark")
    ap.add_argument("--controller", default="pid", help="pid, hold, sac, or module:Class")
    ap.add_argument("--checkpoint")
    ap.add_argument("--scenarios", nargs="+", default=list(BENCHMARK_SCENARIOS),
                    choices=sorted(SCENARIOS))
    ap.add_argument("--cycles", nargs="+", default=list(CYCLES), choices=list(CYCLES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--shield", action="store_true", help="wrap the controller in the Simplex safety shield")
    ap.add_argument("--rods-only", action="store_true", help="disable boron control")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpp", "python"])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", help="write per-episode results to this JSON file")
    args = ap.parse_args(argv)

    label = args.controller + ("+shield" if args.shield else "")
    rows = evaluate(args.controller, args.scenarios, args.cycles, args.seeds, args.shield,
                    args.checkpoint, args.backend, args.hours, not args.rods_only)
    print()
    print(markdown_table(label, rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        meta = {"controller": label, "checkpoint": args.checkpoint, "norman_version": __version__,
                "python": platform.python_version(), "boron_control": not args.rods_only}
        Path(args.out).write_text(json.dumps({"meta": meta, "episodes": rows}, indent=2))
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
