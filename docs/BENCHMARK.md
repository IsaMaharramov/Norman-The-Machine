# Benchmark specification

`Norman/LoadFollow-v1` is a constrained control task. The controller must make reactor
power follow a grid demand profile for 24 hours, using control rods and soluble
boron, while the plant fights back with temperature feedback and xenon poisoning.

## Episode

- **Control interval:** 60 s of plant time per `step()` (`control_interval`).
- **Horizon:** 24 h = 1440 steps (`hours`).
- **Initial state:** exact equilibrium at the profile's first power level: precursors,
  temperatures, iodine, xenon and samarium at steady state, rods at 25% insertion,
  boron at the cycle's critical concentration.

## Actions

`Box(-1, 1, shape=(2,))`. Positive values always add reactivity.

| Index | Meaning | Full scale |
|---|---|---|
| 0 | rod speed: +1 withdraw, −1 insert | 1/180 of full stroke per second |
| 1 | boron rate: +1 dilute, −1 borate | 3 ppm/min |

With `boron_control=False` the action space is `Box(-1, 1, shape=(1,))`, rods only.

## Observations

`Box(shape=(14,))`, roughly O(1):

| Index | Name | Definition |
|---|---|---|
| 0 | `power` | measured power, fraction of rated (`observation_noise` std added) |
| 1 | `target_next` | demand at the end of this step: the value the reward uses |
| 2–4 | `target_15min`, `target_30min`, `target_60min` | dispatch schedule preview |
| 5 | `rod_insertion` | 0 = fully withdrawn, 1 = fully inserted |
| 6 | `boron_kppm` | boron / 1000 ppm |
| 7 | `fuel_temp_dev_100K` | $(T_f - T_{f,0})/100$ |
| 8 | `coolant_temp_dev_10K` | $(T_c - T_{c,0})/10$ |
| 9 | `xenon` | Xe-135, fraction of full-power equilibrium |
| 10 | `iodine` | I-135, fraction of full-power equilibrium |
| 11 | `reactivity_dollars` | total reactivity / β |
| 12 | `power_rate_5pct_min` | last-step power change, in units of 5%/min |
| 13 | `episode_progress` | t / horizon |

Xenon, iodine and reactivity are not measured directly in a plant, but core-monitoring
systems estimate them online. That is why they are part of the observation.

## Reward and cost

Reward per step:

$$r = -\min\left(\left(\frac{P - P_{target}}{0.05}\right)^2, 1\right) - 0.005\,(|a_0| + |a_1|)$$

Cost per step (`info["cost"]`) is 1 if any operating limit was broken during the step:

| Limit | Value | Meaning |
|---|---|---|
| `overpower` | peak power inside the step > 105% | thermal margin |
| `ramp_rate` | \|ΔP\| > 5% of rated per minute | EUR manoeuvrability limit |
| `fuel_temperature` | average fuel > 690 °C | fuel thermal margin |

Each violation type is also reported in `info["violations"]`, so you can use this
task with constrained-RL methods (Lagrangian SAC/PPO, CPO, safety layers) the same way
as Safety-Gymnasium tasks.

**Reactor trip.** If peak power inside a step exceeds 109%, the reactor scrams and the
episode terminates. The trip reward equals the discounted value of losing every
remaining step, $-(1-\gamma^{k})/(1-\gamma)$ with $\gamma = 0.99$. A controller
therefore cannot raise its return by tripping early.

## Scenarios

| Name | Profile |
|---|---|
| `eur_daily` | EUR "12-3-6-3": 12 h at 100%, 3 h ramp to 50%, 6 h at 50%, 3 h ramp back |
| `fast_ramp` | 100 → 50% at 5%/min, 4 h hold, 50 → 100% at 5%/min, then the xenon transient |
| `duck_curve` | Net load after solar and wind: midday dip, steep evening ramp, wind noise |
| `frequency_response` | 90–95% base with secondary frequency-control set-point steps of ±5% |
| `random` | Random holds and ramps (0.5–5%/min) between 50 and 100%, for training |

## Evaluation protocol

A leaderboard entry is the mean over **4 scenarios × 3 cycle states (BOC, MOC, EOC) ×
seeds {0, 1, 2}** = 36 episodes:

```bash
norman evaluate --controller <name> --out results/<name>.json
```

Metrics per episode (`src/norman/evaluation/metrics.py`):

| Metric | Meaning |
|---|---|
| `rmse_pct`, `max_abs_error_pct` | tracking error, % of rated power |
| `time_within_1pct` | fraction of the day within ±1% of demand |
| `energy_deficit_mwh` | thermal energy requested but not delivered |
| `rod_travel_strokes` | total rod travel in full strokes (mechanism wear) |
| `boron_dilution_ppm`, `boration_ppm` | total boron changes (effluent and processing burden) |
| `cost_steps` | steps with any limit violation |
| `trip` | reactor scram |
| `shield_interventions` | steps where the safety shield overrode the controller |

After a trip, power counts as zero for the rest of the horizon in the tracking
metrics.

## Submitting a result

1. Implement a controller with `reset()` and `act(obs) -> np.ndarray`.
2. Run the protocol above and commit `results/<name>.json`.
3. Open a pull request that adds a row to the README table. Describe the method,
   how many plant-days it trained on, and whether it used the shield.
