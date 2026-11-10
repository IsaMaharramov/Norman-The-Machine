from __future__ import annotations

import argparse
import importlib
import sys

from norman import __version__

COMMANDS = {
    "evaluate": ("norman.cli.evaluate", "run the benchmark for a controller"),
    "run": ("norman.cli.run", "run one episode and plot it"),
    "train": ("norman.cli.train", "train the SAC baseline"),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="norman", description="Reactor load-following benchmark")
    parser.add_argument("--version", action="version", version=f"norman {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    modules = {}
    for name, (module_name, help_text) in COMMANDS.items():
        module = importlib.import_module(module_name)
        module.configure(sub.add_parser(name, help=help_text))
        modules[name] = module
    args = parser.parse_args(argv)
    modules[args.command].main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
