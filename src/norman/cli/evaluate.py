from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from norman import __version__
from norman.envs.scenarios import BENCHMARK_SCENARIOS, SCENARIOS
from norman.evaluation import CYCLES, evaluate, markdown_table


def configure(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--controller", default="pid", help="pid, hold, sac, or module:Class")
    ap.add_argument("--checkpoint")
    ap.add_argument("--scenarios", nargs="+", default=list(BENCHMARK_SCENARIOS), choices=sorted(SCENARIOS))
    ap.add_argument("--cycles", nargs="+", default=list(CYCLES), choices=list(CYCLES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--shield", action="store_true", help="wrap the controller in the Simplex safety shield")
    ap.add_argument("--rods-only", action="store_true", help="disable boron control")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpp", "python"])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", help="write per-episode results to this JSON file")


def main(args: argparse.Namespace) -> None:
    label = args.controller + ("+shield" if args.shield else "")
    rows = evaluate(args.controller, args.scenarios, args.cycles, args.seeds, args.shield,
                    args.checkpoint, args.backend, args.hours, not args.rods_only)
    print()
    print(markdown_table(label, rows))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"controller": label, "checkpoint": args.checkpoint, "norman_version": __version__,
                "python": platform.python_version(), "boron_control": not args.rods_only}
        out.write_text(json.dumps({"meta": meta, "episodes": rows}, indent=2))
        print(f"\nresults written to {out}")
