# Plant model

Norman simulates a generic 3 GWth four-loop pressurized water reactor (PWR) with a
reduced-order (point) model. It is built to be **transparent and verifiable**: every
equation is listed here, every parameter has a source or a stated rationale, and the
test suite checks the numerics against analytic solutions.

> **Scope.** This is a control-research model, not a licensing or safety-analysis
> code. It has been *verified* (the equations are solved correctly) but not
> *validated* against plant data. Do not use it to make decisions about a real plant.

## State

| # | Symbol | Meaning | Normalization |
|---|---|---|---|
| 0 | $n$ | fission power | fraction of rated power |
| 1–6 | $c_i$ | delayed-neutron precursors, 6 groups | $c_i = n$ at equilibrium |
| 7 | $T_f$ | average fuel temperature | °C |
| 8 | $T_c$ | average coolant (moderator) temperature | °C |
| 9 | $I$ | iodine-135 | fraction of full-power equilibrium |
| 10 | $X$ | xenon-135 | fraction of full-power equilibrium |
| 11 | $P_m$ | promethium-149 | fraction of full-power equilibrium |
| 12 | $S_m$ | samarium-149 | fraction of full-power equilibrium |

Actuator states: control-rod bank insertion $z \in [0,1]$ (0 = fully withdrawn) and
soluble boron concentration $B$ (ppm).

## Equations

### Point kinetics, six delayed groups

$$\frac{dn}{dt} = \frac{\rho - \beta}{\Lambda}\,n + \sum_{i=1}^{6}\frac{\beta_i}{\Lambda}\,c_i + S
\qquad
\frac{dc_i}{dt} = \lambda_i\,(n - c_i)$$

The precursors are scaled ($c_i \equiv \Lambda\lambda_i C_i/\beta_i$) so that all states
are O(1), which keeps the implicit solver well-conditioned. $S$ is a small intrinsic
neutron source. It keeps a deeply subcritical core at a physical, positive power level,
as in a real reactor.

### Lumped thermal-hydraulics (fuel and coolant nodes)

$$C_f\frac{dT_f}{dt} = f_f P_0 n - hA\,(T_f - T_c)$$

$$C_c\frac{dT_c}{dt} = (1-f_f) P_0 n + hA\,(T_f - T_c) - 2\dot m c_p\,(T_c - T_{in})$$

$hA$ and $\dot m c_p$ come from the nominal full-power temperatures, so the
full-power steady state is exactly $T_f = T_{f,0}$, $T_c = T_{c,0}$.

### Fission-product poisons

With $\varphi = n\,\varphi_0$, the normalized iodine–xenon and promethium–samarium chains are

$$\frac{dI}{dt} = \lambda_I (n - I)$$

$$\frac{dX}{dt} = \frac{\gamma_X}{\gamma}(\lambda_X + \sigma_X\varphi_0)\,n + \frac{\gamma_I}{\gamma}(\lambda_X + \sigma_X\varphi_0)\,I - (\lambda_X + \sigma_X\varphi_0 n)\,X$$

$$\frac{dP_m}{dt} = \lambda_{Pm}(n - P_m)
\qquad
\frac{dS_m}{dt} = \sigma_{Sm}\varphi_0\,(P_m - n\,S_m)$$

with $\gamma = \gamma_I + \gamma_X$. These follow from the usual number-density equations
divided by their full-power equilibrium values. The macroscopic fission cross-section
cancels out.

### Reactivity

$$\rho = \rho_{bias} + \rho_{rod}(z) + \alpha_B B + \alpha_f (T_f - T_{f,0}) + \alpha_m (T_c - T_{c,0}) - w_X X - w_{Sm} S_m$$

- Rod bank integral worth follows the standard S-curve:
  $\rho_{rod}(z) = -W\left(z - \frac{\sin 2\pi z}{2\pi}\right)$.
- Equilibrium poison worths come from the fission yields:
  $w_X = \frac{\gamma}{\nu}\frac{\sigma_X\varphi_0}{\lambda_X + \sigma_X\varphi_0}$ (≈ 2160 pcm) and
  $w_{Sm} = \gamma_{Pm}/\nu$ (≈ 465 pcm).
- $\rho_{bias}$ represents the fuel's excess reactivity. At every reset it is solved
  so that the chosen initial state is exactly critical (a critical-boron search).

### Actuators

Rods move at most $v_{max}$ = 1/180 of full stroke per second (full stroke in 3 min).
Boron changes at most 3 ppm/min by dilution or boration and cannot go below 0 ppm.
That floor is why end-of-cycle load following is hard: there is almost no boron left
to dilute.

## Parameters (MOC defaults)

| Parameter | Value | Source / rationale |
|---|---|---|
| Delayed-group fractions $\beta_i/\beta$ | 0.033, 0.219, 0.196, 0.395, 0.115, 0.042 | Keepin (1965), U-235 thermal fission |
| Group decay constants $\lambda_i$ | 0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01 s⁻¹ | Keepin (1965) |
| $\beta$ (BOC / MOC / EOC) | 0.0065 / 0.0058 / 0.0052 | Falls with Pu-239 build-up; typical PWR range |
| $\Lambda$ (BOC / MOC / EOC) | 22 / 20 / 17 µs | Typical LWR prompt generation time |
| Rated thermal power $P_0$ | 3000 MW | Generic four-loop PWR |
| $T_{in}$, $T_{c,0}$, $T_{f,0}$ | 292, 308, 650 °C | Typical cold-leg, core-average coolant, volume-average fuel |
| Fuel power fraction $f_f$ | 0.974 | Fraction of fission energy deposited in fuel |
| Fuel heat capacity $C_f$ | 4.27e7 J/K | ≈ 100 t UO₂; gives a fuel time constant of ≈ 5 s |
| Coolant heat capacity $C_c$ | 1.2e8 J/K | In-core water inventory; ≈ 0.6 s core transit |
| Doppler $\alpha_f$ (BOC / MOC / EOC) | −2.2 / −2.5 / −2.9 pcm/K | Typical PWR range |
| Moderator $\alpha_m$ (BOC / MOC / EOC) | −12 / −25 / −45 pcm/K | Becomes more negative as boron falls |
| Boron worth $\alpha_B$ (BOC / MOC / EOC) | −7 / −8 / −9 pcm/ppm | Typical PWR range |
| Initial boron (BOC / MOC / EOC) | 1200 / 650 / 40 ppm | Representative let-down curve |
| Control-bank worth $W$ | 2500 pcm | Combined control banks used for manoeuvring |
| Nominal thermal flux $\varphi_0$ | 3e13 n/cm²/s | Typical PWR core average |
| $\gamma_I$, $\gamma_X$, $\gamma_{Pm}$ | 0.0639, 0.00237, 0.0113 | Lamarsh & Baratta |
| $\lambda_I$, $\lambda_X$, $\lambda_{Pm}$ | 2.87e-5, 2.09e-5, 3.63e-6 s⁻¹ | Lamarsh & Baratta |
| $\sigma_X$, $\sigma_{Sm}$ | 2.65e6 b, 4.08e4 b | 2200 m/s values |
| $\nu$ | 2.43 | U-235 thermal fission |
| Neutron source $S$ | 5e-7 s⁻¹ | Puts a −1000 pcm subcritical core near 1e-9 of rated power |

All values are representative open-literature numbers, not data for any specific plant.
`norman.physics.params.sample_cycle` draws a cycle state and perturbs the feedback
coefficients, rod worth and generation time by ±10%, to test controller robustness.

## Numerical method

The system is stiff: prompt-neutron time constants are milliseconds, while xenon
evolves over tens of hours. The C++ core (`cpp/src/reactor.cpp`) integrates it with the
two-stage, L-stable, second-order SDIRK method of Alexander (1977), using full Newton
iterations and the analytic Jacobian. The default inner step is 0.25 s, and a step
whose Newton iteration does not converge is halved automatically. Rod and boron
positions are integrated exactly inside each step, including the moment they hit
their limits. One simulated day at a 1-minute control interval runs in about one
second on a laptop.

The pure-Python backend (`norman/physics/reference.py`) solves the same equations
with SciPy's Radau IIA (5th order, adaptive, rtol 1e-8). It shares no numerical code
with the C++ core, so the agreement between the two is a real check.

## Verification

`pytest` runs these checks on both backends:

| Test | Reference |
|---|---|
| Equilibrium stays steady for one hour at 50% and 100% power | Exact steady state |
| Analytic Jacobian matches finite differences | Central differences |
| C++ and Python right-hand sides agree | Agreement to round-off |
| C++ SDIRK2 transient matches SciPy Radau over 30 min of random rod/boron moves | Difference < 1e-4 |
| Prompt jump after a 0.2$ step | $n_+/n_- = \beta/(\beta-\rho)$, within 1% |
| Stable period after +20 and +100 pcm steps | Root of the inhour equation, within 0.5% |
| Xenon after shutdown: whole curve, peak value and peak time | Closed-form Bateman solution, within 1% |
| Rod worth curve is monotonic with the right total | S-curve definition |

## Known limitations

- **Point model.** There is no spatial flux shape, so no axial offset control and no
  spatial xenon oscillations. A 1-D axial model is the most important next step.
- **No secondary side.** Inlet temperature is fixed. There is no steam generator,
  turbine or $T_{avg}$ program, so demand is a reactor-power set-point.
- **Single lumped rod bank.** Real plants move several overlapping banks with
  step-wise motion and deadbands.
- **Boron acts instantly in the core.** Mixing and transport delays are ignored.
- **Representative parameters.** Coefficients describe a generic PWR, not a specific
  core design or burnup state.

## References

- G. R. Keepin, *Physics of Nuclear Kinetics*, Addison-Wesley, 1965.
- J. R. Lamarsh, A. J. Baratta, *Introduction to Nuclear Engineering*, 3rd ed., Prentice Hall, 2001.
- J. J. Duderstadt, L. J. Hamilton, *Nuclear Reactor Analysis*, Wiley, 1976.
- D. L. Hetrick, *Dynamics of Nuclear Reactors*, University of Chicago Press, 1971.
- R. Alexander, "Diagonally implicit Runge–Kutta methods for stiff ODEs", *SIAM J. Numer. Anal.* 14(6), 1977.
- OECD/NEA, *Technical and Economic Aspects of Load Following with Nuclear Power Plants*, 2011.
- European Utility Requirements for LWR Nuclear Power Plants (EUR), manoeuvrability requirements.
