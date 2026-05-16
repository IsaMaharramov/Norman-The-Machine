# Norman The Machine
**Autonomous Nuclear Reactor Load-Following System via Recurrent Soft Actor-Critic (SAC)**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Active%20Training-success.svg)

> ⚠️ **DRAFT / ACTIVE RESEARCH:** This repository is currently undergoing active development. The C++ physics bindings, LSTM tensor dimensionalities, and mathematical reward formulations are being actively tuned. Code and documentation are subject to frequent changes.

## Abstract
**Norman_The_Machine** is a high-performance, physics-aware Reinforcement Learning framework designed to solve the highly non-linear control problem of nuclear reactor load-following. The system bridges a deterministic C++ nuclear kinetics engine (using 4th-Order Runge-Kutta integration) with a stochastic optimal control AI (PyTorch Recurrent Soft Actor-Critic). The primary optimization objective is to dynamically track highly variable grid energy demands while strictly preventing Xenon-135 induced "poison-out" transients.

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

## Machine Learning Architecture: Recurrent SAC
Standard Markovian Reinforcement Learning agents fail in nuclear control because they cannot observe the derivative of the Iodine concentration. To solve the Xenon time-lag, this project implements a **Recurrent Soft Actor-Critic (LSTM-SAC)** architecture.

### The Soft Actor-Critic Objective
SAC is an off-policy algorithm that optimizes a stochastic policy in an entropy-regularized framework. This prevents the control rods from converging to sub-optimal, rigid policies during early training. The objective is to maximize both expected return and entropy $\mathcal{H}$:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

### Twin-Q Critic Network with LSTM Embedding
To capture the historical trajectory of the reactor state, both the Actor and the Twin-Critics utilize a Long Short-Term Memory (LSTM) backbone. The observation state $s_t$ is embedded into a hidden state $h_t$:

$$h_t = \text{LSTM}(s_t, h_{t-1})$$

The Critics minimize the Mean Squared Bellman Error (MSBE) using the temporal embedding, mitigating the overestimation bias inherent in continuous-action Q-learning:

$$J_Q(\phi_i) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_{\phi_i}(h, a) - \left(r + \gamma \mathbb{E}_{s'}[V_{\bar{\phi}}(h')] \right) \right)^2 \right] \quad \text{for } i \in \{1, 2\}$$

### Feature Scaling & Numerical Hardening
To maintain gradient stability during backpropagation, physical values ranging from $10^{13}$ to $10^{16}$ are continuously transformed via log-scaling before entering the neural network.

$$obs = \left[ \log_{10}(\phi + \epsilon), \log_{10}(I + \epsilon), \log_{10}(X + \epsilon), P_{current}, P_{target} \right]$$

---

## Optimization Surface (Reward Formulation)
The agent navigates a highly non-linear optimization surface. The reward function is dense and strictly penalizes both grid deviation and unsafe transient states. It is composed of three distinct penalty mechanisms to ensure smooth and safe load-following:

**1. Quadratic Accuracy Penalty:**
To enforce tight load-following tolerances, deviations from the target grid demand are penalized quadratically. This forces the agent to aggressively correct large errors while allowing precise micro-adjustments near the target.
$$R_{accuracy} = -30.0 \cdot (P_{actual} - P_{target})^2$$

**2. Advanced Exponential Safety Barrier:**
To prevent gradient starvation while strictly enforcing the Poison-Out limit ($5 \times 10^{16}$), the Xenon penalty is formulated as an exponential barrier. This allows precise load-following in safe states while creating an insurmountable mathematical wall near critical thresholds.
$$R_{safety} = -5.0 \cdot \left(e^{6.0 \cdot \left(\frac{X_{actual}}{5 \times 10^{16}}\right)} - 1.0\right)$$

**3. Critical Failure Guard:**
If the agent's initial random exploration induces unrecoverable numerical stiffness in the C++ ODE solver (e.g., $P_{actual}$ approaching infinity or `NaN`), the episode is immediately truncated with a catastrophic penalty. This quarantines the mathematical instability, preventing broken data from entering the Replay Buffer and corrupting the network weights.
$$R_{critical} = -500.0$$

**Total Reward Function:**
$$R_t = R_{accuracy} + R_{safety} + R_{critical}$$

*(Note: Hard episode termination occurs if $X_{actual} > 5 \times 10^{16}$)*

---

## System Architecture
```text
NORMAN_THE_MACHINE/
├── agent/
│   ├── networks.py       # PyTorch LSTM Actor & Twin-Critic classes
│   └── sac_agent.py      # Optimization logic, Bellman updates, Replay Buffer
├── core/
│   ├── include/          # C++ Headers (ReactorDynamics.hpp)
│   └── src/              # C++ RK4 Integrator & Pybind11 bindings
├── data/                 # Saved model weights (.pth)
├── env/
│   ├── demand.py         # Grid load-following trajectory generator
│   └── reactor_env.py    # Gymnasium Wrapper with numerical hardening
├── scripts/
│   └── dashboard.py      # Matplotlib real-time telemetry visualization
├── CMakeLists.txt        # C++ build configuration
└── train.py              # Main training loop and memory management
```

---

## Installation

### 1. Install Requirements:

```bash
pip install -r requirements.txt
```

### 2. Compile the Physics Core:

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 3. Deploy the Bridge:

Copy the compiled `norman_core.*.pyd` (or `.so`) file from the `build/Release` directory into the project root.

---

## Training the Model
Initialize the Recurrent SAC agent and begin the load-following simulation:

```bash
python train.py
```