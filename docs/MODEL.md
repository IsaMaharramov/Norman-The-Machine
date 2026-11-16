# Plant model

Norman simulates a generic 3 GWth four-loop pressurized water reactor (PWR) with a
reduced-order model: point kinetics for the fast dynamics and, optionally, a 1-D axial
core for the power shape and xenon redistribution. It is built to be **transparent
and verifiable**: every
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
  $w_X = \frac{\gamma}{\nu}\frac{\sigma_X\varphi_0}{\lambda_X + \sigma_X\varphi_0}$ (≈ 2360 pcm) and
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
| Nominal thermal flux $\varphi_0$ | 5e13 n/cm²/s | Core-average thermal flux; gives an equilibrium xenon worth of 2360 pcm and a 25 h axial xenon oscillation period |
| $\gamma_I$, $\gamma_X$, $\gamma_{Pm}$ | 0.0639, 0.00237, 0.0113 | Lamarsh & Baratta |
| $\lambda_I$, $\lambda_X$, $\lambda_{Pm}$ | 2.87e-5, 2.09e-5, 3.63e-6 s⁻¹ | Lamarsh & Baratta |
| $\sigma_X$, $\sigma_{Sm}$ | 2.65e6 b, 4.08e4 b | 2200 m/s values |
| $\nu$ | 2.43 | U-235 thermal fission |
| Neutron source $S$ | 5e-7 s⁻¹ | Puts a −1000 pcm subcritical core near 1e-9 of rated power |

All values are representative open-literature numbers, not data for any specific plant.
`norman.physics.params.sample_cycle` draws a cycle state and perturbs the feedback
coefficients, rod worth and generation time by ±10%, to test controller robustness.

## Axial core (`core="axial"`)

The point model knows how much power the core makes, not *where*. The axial model
adds the spatial dimension that matters most for load following: the 3.66 m core
height, along which xenon can redistribute and drive slow **axial xenon
oscillations**.

### Method: quasi-static shape + point amplitude

The core is split into $N = 40$ axial nodes. Each node $i$ carries its own fuel and
coolant temperature and its own I-135, Xe-135, Pm-149 and Sm-149 inventories. Power is
factorized as $P_i(t) = n(t)\,\psi_i(t)$:

- **Amplitude $n(t)$:** the existing point-kinetics model, with its 6 delayed groups
  and dynamic thermal feedback on 0.25 s steps.
- **Shape $\psi_i$:** recomputed every coupling interval (5 s) from a one-group 1-D
  diffusion eigenvalue problem, with all slow axial physics folded into the local
  reactivity.

$$-M^2\frac{d^2\phi}{dz^2} + \phi = \frac{1}{k}\,\frac{k_{\infty,0}}{1 - \rho_i}\,\phi$$

$$\rho_i = b + \tau\left(\tfrac{z_i}{H} - \tfrac12\right) - W\,c_i + \alpha_B B + \alpha_f (T_{f,i} - T_{f,0}) + \alpha_m (T_{c,i} - T_{c,0}) - w_X X_i - w_{Sm} S_{m,i}$$

- $c_i$ is the fraction of node $i$ covered by the rod bank, which enters **from the
  top**. With a uniform absorber per covered node, the S-shaped integral rod worth
  curve is no longer an input. It emerges from the flux shape: small at both ends,
  steepest mid-core, totalling $W$.
- **Coolant heats as it flows up.** The inlet of each node is the outlet of the node
  below, so the moderator is coldest at the bottom. Local temperatures are the point
  model's *current, lagged* core-average temperatures distributed by the flux shape:
  the coolant rise uses the thermal power $n_c = (T_c - T_{in})/(T_{c,0} - T_{in})$
  and the fuel-to-coolant difference uses $n_f = (T_f - T_c)/(T_{f,0} - T_{c,0})$. At
  steady state $n_c = n_f = n$. In a transient, the spatial model sees the same thermal
  lag as the amplitude (see the coupling section for why this matters).
- Local poisons follow the point-model equations driven by local flux
  $n\,\psi_i\,\varphi_0$. They are integrated across the step with exact exponentials
  (I, Pm) and the trapezoidal rule (Xe, Sm).
- $k_{\infty,0}$ is set so that $\rho_i \equiv 0$ gives $k = 1$. For a uniform $\rho$
  the global reactivity is exactly $\rho$ (tested).
- **Burnup tilt $\tau$.** Fresh, uniform fuel would put the power strongly at the
  bottom, because the coolant is coldest there. Real cores are flattened by their
  burnup distribution. A linear reactivity tilt $\tau$ stands in for that. It is
  calibrated once per episode so that the full-power equilibrium axial offset equals a
  design value (−3%). The uniform bias $b$ is then solved for criticality, as in the
  point model.

### Coupling

Each 60 s control step is split into coupling intervals of $\Delta t_c$ = 5 s. In each
interval the point model integrates the fast dynamics, the axial poisons advance, and
the shape is re-solved. The eigenvalue reactivity $\rho_{eig}$ then replaces the point
model's reactivity through a correction to its bias,
$\Delta\rho = \rho_{eig} - \rho_{point}$. Two design decisions make this coupling
stable and accurate. Both were found by testing, not assumed.

1. **Live rod worth.** Inside an interval the rods keep moving, at up to a third of
   the stroke per minute. Their real worth depends on the flux shape: with power
   piled at the top, the top of the stroke is worth far more than the textbook
   S-curve says. At the start of every control step the axial core therefore computes
   the full integral rod worth curve for its current state (21 eigenvalue solves with
   frozen temperatures and poisons), and the point model uses that table inside the
   step. Without it, the correction absorbed rod-worth errors of up to 600 pcm as
   step-wise reactivity jumps, which drove a classical controller into a limit cycle
   and a spurious trip.
2. **Lagged local temperatures.** An earlier version made local temperatures follow
   local power instantly. That injects instantaneous power feedback, through the
   correction, into neutron kinetics that respond in milliseconds. Applied with a
   sampling delay, such a loop is unstable: an undisturbed reactor diverged at
   coupling intervals below 5 s. With local temperatures built from the point model's
   lagged thermal powers $n_c, n_f$, the plant holds equilibrium to 1e-9 at any
   interval from 60 s to 1 s (tested).

If the amplitude settles within an interval, the coupling maps a power error $e_k$ to
$e_{k+1} = (1 - a/b)\,e_k$, where $a$ and $b$ are the axial and point power
coefficients. It converges whenever $0 < a/b < 2$. Measured ratios are 1.22 / 1.20 /
1.20 (BOC / MOC / EOC). The axial coefficient is larger because flux-squared weighting
emphasizes the hot, peaked mid-core.

**Convergence.** Replaying the same full-day action sequences with coupling intervals
of 60, 20, 10 and 5 s against a 2.5 s reference shows first-order convergence: the
error halves with the interval. At 5 s the worst deviation over the most violent
scenario (`fast_ramp`) is about 1% of rated power and 0.3% in axial offset. At 60 s
it reaches 12–18% of power, which is why the default is 5 s
(`AxialPlant(coupling_interval=...)`, `LoadFollowEnv(coupling_interval=...)`).

### Discretization and eigensolver

Finite volumes of height $h = H/N$, with a Robin condition $\phi'/\phi = 1/\delta$ at
both core faces ($\delta$ = 8 cm reflector savings). The generalized problem
$A\phi = \lambda F\phi$ is symmetrized to $F^{-1/2}AF^{-1/2}$, a symmetric tridiagonal
matrix. Its smallest eigenvalue is found by Sturm-sequence bisection, and the
eigenvector by shifted inverse iteration. The method is exact to round-off, needs no
initial guess, and takes about 60 µs per step in C++. The Python reference uses LAPACK
(`scipy.linalg.eigh_tridiagonal`) and an exact matrix exponential for the poisons, so
it shares no numerical method with the C++ core.

### Axial parameters

| Parameter | Value | Source / rationale |
|---|---|---|
| Active core height $H$ | 366 cm | 12 ft, standard four-loop PWR |
| Nodes $N$ | 40 | 9.15 cm each; eigenvalue converges at 2nd order |
| Reflector savings $\delta$ | 8 cm | Typical light-water reflector |
| Migration area $M^2$ | 50 cm² | One-group $L^2 + \tau$ for a PWR lattice; sets axial stability |
| Design axial offset | −3% | Typical full-power target |

$H/M \approx 52$: the core is about 52 migration lengths tall. That is what makes
large PWRs vulnerable to axial xenon oscillations. A smaller $M^2$ or a higher flux
makes them less damped.

### Axial xenon stability

The standard test: perturb the power shape with a rod pulse, hold total power constant
with boron, and watch the axial offset. With the defaults:

| Cycle | Period | Decay ratio per cycle |
|---|---|---|
| BOC | 24.7 h | 0.30 |
| MOC | 25.2 h | 0.21 |
| EOC | 26.0 h | 0.13 |

The period sits in the 24–30 h range reported for large PWRs. The oscillations are
damped but last several cycles, so an uncontrolled load change visibly sloshes power
up and down the core. With $M^2$ = 40 cm² the decay ratio rises to 0.3–0.8, closer to
the stability limit. The environment accepts any `AxialParams`, so stability studies
are one argument away.

### Operating limit: ΔI band

PWRs following load keep $\Delta I = \text{AO} \times P$ within a band around its
target (constant axial offset control). The axial environment adds an `axial_offset`
violation when $|\Delta I - \Delta I_{target}|$ exceeds 5% of rated power, with
$\Delta I_{target} = \text{AO}_{design} \times P$.

## Numerical method

The system is stiff: prompt-neutron time constants are milliseconds, while xenon
evolves over tens of hours. The C++ core (`cpp/src/reactor.cpp`) integrates it with the
two-stage, L-stable, second-order SDIRK method of Alexander (1977), using full Newton
iterations and the analytic Jacobian. The default inner step is 0.25 s, and a step
whose Newton iteration does not converge is halved automatically. Rod and boron
positions are integrated exactly inside each step, including the moment they hit
their limits. One simulated day at a 1-minute control interval runs in about one
second on a laptop.

The pure-Python backend (`src/norman/physics/reference.py`) solves the same equations
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
| Axial: uniform local reactivity gives exactly that global reactivity | Exact |
| Axial: fundamental mode matches the diffusion cosine | Analytic, within 2e-3 |
| Axial: leakage eigenvalue converges at 2nd order to the Robin-condition solution | $B\tan(BH/2) = 1/\delta$ |
| Axial: Sturm bisection matches LAPACK | Agreement to 1e-13 |
| Axial: C++ core matches the Python reference over 1 h of transient | Difference < 1e-5 |
| Axial: rod worth emerges as an S-curve with total $W$ | Within 2% |
| Axial: initialization hits the design offset and criticality; equilibrium is steady | Exact |
| Axial: xenon oscillation period after a rod pulse, BOC/MOC/EOC | 20–32 h, decaying |
| Axial: coolant energy balance at 60% and 100% power | Exact outlet temperature |
| Axial: coupling holds equilibrium at intervals of 60, 5 and 1 s | Steady to 1e-7 |
| Axial: coupling converges as the interval shrinks | Error < 2% at 5 s, decreasing |
| Axial: rod worth table, C++ vs Python | Agreement to 1e-12 |

## Known limitations

- **One-group, one-dimensional shape.** No radial dimension, no energy groups, and
  one lumped rod bank. Burnup appears only as a linear axial tilt calibrated to a
  design offset.
- **Quasi-static shape.** The shape is re-solved every 5 s with first-order coupling
  accuracy. That is accurate for xenon- and rod-driven shape changes, but not for
  sub-second spatial transients such as a rod ejection.
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
