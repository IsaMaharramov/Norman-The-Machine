from __future__ import annotations

import math

IF97_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5,
    -0.32325550322333e7, 0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6,
    -0.23855557567849, 0.65017534844798e3,
)


def saturation_pressure(t_celsius: float) -> float:
    n = IF97_N
    t = t_celsius + 273.15
    theta = t + n[8] / (t - n[9])
    a = theta * theta + n[0] * theta + n[1]
    b = n[2] * theta * theta + n[3] * theta + n[4]
    c = n[5] * theta * theta + n[6] * theta + n[7]
    return (2.0 * c / (-b + math.sqrt(b * b - 4.0 * a * c))) ** 4


def saturation_pressure_slope(t_celsius: float, h: float = 1e-4) -> float:
    return (saturation_pressure(t_celsius + h) - saturation_pressure(t_celsius - h)) / (2.0 * h)
