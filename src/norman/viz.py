from __future__ import annotations

import numpy as np


def _power_panel(ax, t, history, title):
    power = np.array([h["power"] for h in history]) * 100
    target = np.array([h["target"] for h in history]) * 100
    cost = np.array([h["cost"] for h in history])
    ax.plot(t, target, "--", color="#d9822b", lw=1.6, label="grid demand")
    ax.plot(t, power, color="#1f6fb2", lw=1.4, label="reactor power")
    bad = cost > 0
    if bad.any():
        ax.scatter(t[bad], power[bad], color="#c0392b", s=8, zorder=5, label="limit violated")
    ax.set_ylabel("power [% rated]")
    ax.legend(loc="lower left", ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)


def _actuator_panel(ax, t, history):
    rods = np.array([h["rod"] for h in history]) * 100
    boron = np.array([h["boron"] for h in history])
    ax.plot(t, rods, color="#2e8b57", lw=1.4, label="rod insertion")
    ax.set_ylabel("rod insertion [%]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, boron, color="#8e44ad", lw=1.2, label="boron")
    ax2.set_ylabel("boron [ppm]")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", frameon=False)


def _reactivity_panel(ax, t, history):
    rho0 = history[0]["rho_pcm"]
    colors = {"rods": "#2e8b57", "boron": "#8e44ad", "doppler": "#e67e22", "moderator": "#3498db",
              "xenon": "#c0392b", "samarium": "#7f8c8d", "shape": "#34495e"}
    for k in rho0:
        if k == "bias":
            continue
        v = np.array([h["rho_pcm"][k] - rho0[k] for h in history])
        ax.plot(t, v, lw=1.3, label=k, color=colors.get(k))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("reactivity change [pcm]")
    ax.legend(loc="upper left", ncol=7, frameon=False, fontsize=8)
    ax.grid(alpha=0.3)


def _poison_panel(ax, t, history):
    ax.plot(t, [h["xenon"] for h in history], color="#c0392b", lw=1.4, label="Xe-135")
    ax.plot(t, [h["iodine"] for h in history], color="#f1c40f", lw=1.4, label="I-135")
    ax.set_ylabel("fraction of full-power eq.")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)


def _delta_i_panel(ax, t, history, band):
    delta_i = np.array([h["delta_i"] for h in history]) * 100
    target = np.array([h["delta_i_target"] for h in history]) * 100
    ax.fill_between(t, target - band * 100, target + band * 100, color="#2ecc71", alpha=0.18,
                    label="allowed band")
    ax.plot(t, target, "--", color="#27ae60", lw=1.0)
    ax.plot(t, delta_i, color="#1f6fb2", lw=1.4, label="ΔI (axial offset × power)")
    ax.set_ylabel("ΔI [% rated]")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    ax.grid(alpha=0.3)


def _profile_panel(ax, fig, t, history, key, label, cmap, relative):
    data = np.array([h[key] for h in history]).T
    if relative:
        data = data / data[:, :1]
    mesh = ax.pcolormesh(t, np.linspace(0, 100, data.shape[0]), data, shading="auto", cmap=cmap)
    ax.set_ylabel("core height [%]")
    fig.colorbar(mesh, ax=ax, pad=0.01, label=label)


def plot_episode(history: list[dict], title: str = "", path: str | None = None, show: bool = False,
                 axial_band: float = 0.05):
    import matplotlib

    if path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([h["t_min"] for h in history]) / 60.0
    axial = "shape" in history[0]

    if axial:
        fig, axs = plt.subplots(6, 1, figsize=(12, 16), sharex=True, constrained_layout=True)
        _power_panel(axs[0], t, history, title)
        _actuator_panel(axs[1], t, history)
        _delta_i_panel(axs[2], t, history, axial_band)
        _profile_panel(axs[3], fig, t, history, "shape", "relative power", "inferno", False)
        _profile_panel(axs[4], fig, t, history, "xenon_profile", "Xe-135 / initial", "RdBu_r", True)
        _reactivity_panel(axs[5], t, history)
    else:
        fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
        _power_panel(axs[0], t, history, title)
        _actuator_panel(axs[1], t, history)
        _reactivity_panel(axs[2], t, history)
        _poison_panel(axs[3], t, history)
    axs[-1].set_xlabel("time [h]")

    if path:
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
