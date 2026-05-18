# Norman The Machine
**Autonomous Nuclear Reactor Load-Following System via Soft Actor-Critic (SAC)**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## Abstract
**Norman_The_Machine** is a high-performance, physics-aware Reinforcement Learning framework designed to solve the highly non-linear control problem of nuclear reactor load-following. The system bridges a deterministic C++ nuclear kinetics engine (using 4th-Order Runge-Kutta integration) with a stochastic optimal control AI (PyTorch Soft Actor-Critic). The primary optimization objective is to dynamically track highly variable grid energy demands while strictly preventing Xenon-135 induced "poison-out" transients.

---

## The Physics Core: Deterministic Kinetics
The simulation environment is powered by a custom C++ module (`norman_core`), bound to Python via Pybind11. It solves the Point Kinetics Equations (PKE) coupled with Iodine-Xenon decay dynamics.

### Iodine-Xenon Dynamics
The dominant constraint in reactor load-following is the temporal lag between power adjustments and Xenon-135 concentration. The core models this continuous-time dynamic system:

**Iodine-135 Production:**
$$\frac{dI(t)}{dt} = \gamma_I \Sigma_f \phi(t) - \lambda_I I(t)$$

**Xenon-135 Concentration:**
$$\frac{dX(t)}{dt} = \gamma_X \Sigma_f \phi(t) + \lambda_I I(t) - \lambda_X X(t) - \sigma_X X(t) \phi(t)$$

Because Xenon-135 has a massive neutron absorption cross-section ($\sigma_X$), a rapid decrease in power ($\phi$) causes an accumulation of Xenon (as Iodine continues to decay into it, but fewer neutrons are available to "burn" it away). The RL agent must learn to predict and mitigate this invisible, time-delayed accumulation.

---

## Machine Learning Architecture: Continuous SAC
To map the continuous observation space to real-world control rod positions, this project implements a **Soft Actor-Critic (SAC)** architecture utilizing high-speed Dense Multi-Layer Perceptrons (MLPs).

### The Soft Actor-Critic Objective
SAC is an off-policy algorithm that optimizes a stochastic policy in an entropy-regularized framework. This prevents the control rods from converging to sub-optimal, rigid policies during early training. The objective is to maximize both expected return and entropy $\mathcal{H}$:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

### Action Space Mapping & Feature Scaling
To maintain gradient stability during backpropagation, physical values ranging from $10^{13}$ to $10^{16}$ are continuously transformed via log-scaling before entering the neural network. 

Furthermore, to protect the C++ Runge-Kutta integrator from mathematically fatal negative neutron flux, the SAC's native `tanh` output `[-1.0, 1.0]` is strictly mapped to physical control rod bounds `[0.0, 1.0]` within the Gymnasium step function.

---

## Optimization Surface (Reward Formulation)
The agent navigates a highly non-linear optimization surface. The reward function is dense and strictly penalizes both grid deviation and unsafe transient states. 

**1. Quadratic Accuracy Penalty:**
Enforces tight load-following tolerances, forcing the agent to aggressively correct large errors.
$$R_{accuracy} = -30.0 \cdot (P_{actual} - P_{target})^2$$

**2. Advanced Exponential Safety Barrier:**
To strictly enforce the Xenon Poison-Out limit ($5 \times 10^{16}$), the Xenon penalty is formulated as an exponential barrier. 
$$R_{safety} = -5.0 \cdot \left(e^{6.0 \cdot \left(\frac{X_{actual}}{5 \times 10^{16}}\right)} - 1.0\right)$$

**3. Global Reward Scaling (Anti-Collapse):**
During early exploration, exponential Xenon penalties can yield errors exceeding $-84,000$, resulting in catastrophic PyTorch gradient explosions. To stabilize the Twin-Q Critics, the final environment reward is globally scaled:
$$R_{final} = \frac{R_{accuracy} + R_{safety}}{100.0}$$

---

## Research Outcomes: The Alignment Problem & "Safe Local Optimum"
Following a 2,000-episode training loop, the agent demonstrated a fascinating example of the AI Alignment Problem (Reward Hacking). 

Because the penalty for missing grid demand was quadratic, but the penalty for spiking Xenon was *exponential*, the SAC's Critic networks mathematically deduced that exploring variable power levels was a statistical death trap. Rather than attempting to track the grid and risking a meltdown, the agent discovered a "Safe Local Optimum." 

The trained agent autonomously learned to drop the control rods to $0.0$ and completely shut the reactor down. By doing so, it willingly absorbs a predictable, flat penalty for missing the grid demand (scoring exactly $-189$ per episode), but mathematically guarantees 100% survival by permanently avoiding the Xenon threshold. **The AI prioritized absolute physical safety over industrial efficiency.**

---

## System Architecture
```text
NORMAN_THE_MACHINE/
├── agent/
│   ├── networks.py       # PyTorch Dense MLP Actor & Twin-Critic classes
│   └── sac_agent.py      # Optimization logic, Bellman updates, Replay Buffer
├── core/
│   ├── include/          # C++ Headers (ReactorDynamics.hpp)
│   └── src/              # C++ RK4 Integrator & Pybind11 bindings
├── data/                 # Saved model weights (.pth)
├── envs/
│   ├── demand.py         # Grid load-following trajectory generator
│   └── reactor_env.py    # Gymnasium Wrapper with numerical hardening
├── scripts/
│   └── dashboard.py      # Matplotlib real-time telemetry visualization
├── CMakeLists.txt        # C++ build configuration
└── train.py              # Main training loop and memory management
```

---

## Installation
1. Install Requirements:
```bash
pip install -r requirements.txt
```
2. Compile the Physics Core:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
3. Deploy the Bridge:
```text
Copy the compiled norman_core.*.pyd (or .so) file from the build/Release directory into the project root.
```
---

## Evaluating the Model
To observe the agent's behavior and the real-time Matplotlib telemetry dashboard, load the trained .pth weights in evaluation mode:

```bash
python test.py
```