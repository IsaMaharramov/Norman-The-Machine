from __future__ import annotations

import argparse

from norman.control.shield import SafetyShield
from norman.envs.load_follow import LoadFollowEnv
from norman.envs.scenarios import SCENARIOS
from norman.evaluation import make_controller, run_episode
from norman.viz import plot_episode


def configure(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--controller", default="pid", help="pid, hold, sac, or module:Class")
    ap.add_argument("--checkpoint")
    ap.add_argument("--scenario", default="eur_daily", choices=sorted(SCENARIOS))
    ap.add_argument("--cycle", default="MOC", choices=["BOC", "MOC", "EOC"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--shield", action="store_true")
    ap.add_argument("--rods-only", action="store_true")
    ap.add_argument("--plot", help="save the figure to this path")
    ap.add_argument("--show", action="store_true", help="open an interactive window")


def main(args: argparse.Namespace) -> None:
    env = LoadFollowEnv(scenario=args.scenario, cycle=args.cycle, hours=args.hours,
                        boron_control=not args.rods_only)
    if args.shield:
        env = SafetyShield(env)
    controller = make_controller(args.controller, args.checkpoint, not args.rods_only)
    history, m = run_episode(env, controller, args.seed)

    for k, v in m.items():
        print(f"{k:>22}: {v:.3f}" if isinstance(v, float) else f"{k:>22}: {v}")
    if args.plot or args.show:
        title = f"{args.controller}{' + shield' if args.shield else ''} | {args.scenario} | {args.cycle}"
        plot_episode(history, title, args.plot, args.show)
        if args.plot:
            print(f"figure written to {args.plot}")

