import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("core", ["point", "axial"])
def test_evaluate_writes_results_on_a_legacy_windows_console(tmp_path, core):
    out = tmp_path / "result.json"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    proc = subprocess.run(
        [sys.executable, "-m", "norman", "evaluate", "--core", core, "--scenarios", "eur_daily",
         "--cycles", "MOC", "--seeds", "0", "--hours", "0.5", "--out", str(out)],
        capture_output=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    data = json.loads(out.read_text())
    assert data["meta"]["core"] == core
    assert len(data["episodes"]) == 1


def test_version_command():
    proc = subprocess.run([sys.executable, "-m", "norman", "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.startswith("norman ")
