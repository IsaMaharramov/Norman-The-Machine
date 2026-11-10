from __future__ import annotations

import numpy as np


def plot_episode(history: list[dict], title: str = "", path: str | None = None, show: bool = False):
    import matplotlib

    if path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([h["t_min"] for h in history]) / 60.0
    power = np.array([h["power"] for h in history]) * 100
    target = np.array([h["target"] for h in history]) * 100
    rods = np.array([h["rod"] for h in history]) * 100
    boron = np.array([h["boron"] for h in history])
    rho0 = history[0]["rho_pcm"]
    rho = {k: np.array([h["rho_pcm"][k] - rho0[k] for h in history]) for k in rho0 if k != "bias"}
    xenon = np.array([h["xenon"] for h in history])
    iodine = np.array([h["iodine"] for h in history])
    cost = np.array([h["cost"] for h in history])

    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    ax = axs[0]
    ax.plot(t, target, "--", color="#d9822b", lw=1.6, label="grid demand")
    ax.plot(t, power, color="#1f6fb2", lw=1.4, label="reactor power")
    bad = cost > 0
    if bad.any():
        ax.scatter(t[bad], power[bad], color="#c0392b", s=12, zorder=5, label="limit violated")
    ax.set_ylabel("power [% rated]")
    ax.legend(loc="lower left", ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)

    ax = axs[1]
    ax.plot(t, rods, color="#2e8b57", lw=1.4, label="rod insertion")
    ax.set_ylabel("rod insertion [%]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, boron, color="#8e44ad", lw=1.2, label="boron")
    ax2.set_ylabel("boron [ppm]")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", frameon=False)

    ax = axs[2]
    colors = {"rods": "#2e8b57", "boron": "#8e44ad", "doppler": "#e67e22", "moderator": "#3498db",
              "xenon": "#c0392b", "samarium": "#7f8c8d"}
    for k, v in rho.items():
        ax.plot(t, v, lw=1.3, label=k, color=colors.get(k))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("reactivity change [pcm]")
    ax.legend(loc="upper left", ncol=6, frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[3]
    ax.plot(t, xenon, color="#c0392b", lw=1.4, label="Xe-135")
    ax.plot(t, iodine, color="#f1c40f", lw=1.4, label="I-135")
    ax.set_ylabel("fraction of full-power eq.")
    ax.set_xlabel("time [h]")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)

    if path:
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
