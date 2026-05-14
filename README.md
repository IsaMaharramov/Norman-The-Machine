# My notes?
My notes ~ everything I do


---

# My notes 1:

**Iodine-Xenon Differential Equations**

The engine will solve these two coupled equations every time step:

1. **Iodine Dynamics:**

   $$
   \frac{dI}{dt} = \gamma_I \Sigma_f \Phi - \lambda_I I
   $$

2. **Xenon Dynamics:**

   $$
   \frac{dX}{dt} = (\gamma_X \Sigma_f \Phi + \lambda_I I) - (\lambda_X + \sigma_X \Phi)X
   $$

If the flux $\Phi$ drops too fast (the agent tries to lower power), the $(\lambda_X + \sigma_X \Phi)X$ term becomes small, causing $X$ to spike. This is the "poison" we have to avoid.

---

# My notes 2:

We will implement the physics using a **4th-order Runge-Kutta (RK4) integrator** instead of a simple Euler method. Why? Because nuclear reactor kinetics are "stiff" differential equations;


---

# My notes 3:

core/src/ReactorDynamics.cpp -> physics part

core/src/bindings.cpp -> to bridge C++ and Python -> Pybind11

I update CMakeLists.txt -> tells the compiler how to build the norman_core python module:
```cmake
cmake_minimum_required(VERSION 3.12)
project(Norman_The_Machine)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Define the python module
pybind11_add_module(norman_core core/src/bindings.cpp core/src/ReactorDynamics.cpp)

# Include the headers
target_include_directories(norman_core PRIVATE core/include)
```

## Subnote 1:

**build**

### Subsubnote 1:

I update CMakeLists.txt -> pybind things:

```cmake
cmake_minimum_required(VERSION 3.12)
project(Norman_The_Machine)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

pybind11_add_module(norman_core core/src/bindings.cpp core/src/ReactorDynamics.cpp)

target_include_directories(norman_core PRIVATE core/include)
```

---

# My notes 4:

## subnote 1:

reactor_env.py

## subnote 2:

Long Short-Term Memory (LSTM)

networks.py

## subnote 3:

# Soft Actor-Critic (SAC)

The goal of **SAC** is to maximize the expected reward while also maximizing **Entropy ($H$)**. This ensures the agent remains "curious" and explores the boundaries of reactor stability.

### Actor Objective Function

The objective function for the Actor is:

$$J(\theta) = \mathbb{E}_{s_t \sim D, a_t \sim \pi_{\theta}}[\alpha \log(\pi_{\theta}(a_t|s_t)) - Q_{\phi}(s_t, a_t)]$$

**Where:**

*   **$\alpha$** is the temperature parameter (controlling the trade-off between entropy and reward).
*   **$Q_{\phi}$** is the Critic's estimation of future value.

---

# My notes 5:

## subnote 1:

train.py

## subnote 2:

sac_agent.py

## subnote 3:

```powershell
copy build\Release\norman_core.cp313-win_amd64.pyd .
```

## subnote 4:

updated -> sac_agent.py

The class manages the learning process. It calculates the loss for the Actor and the Critics, ensuring the control rods move in a way that maximizes power accuracy while minimizing the Xenon penalty.

## The Math of Optimization

The Bellman Equation. The core objective -> to minimize the **Mean Squared Bellman Error (MSBE)**:

$$J_Q(\phi) = \mathbb{E}_{(s,a) \sim D} \left[ \frac{1}{2} \left( Q_\phi(s, a) - (r + \gamma \mathbb{E}_{s' \sim P} [V_{\bar{\phi}}(s')]) \right)^2 \right]$$

---

### Actor Update
The Actor is then updated using the **Policy Gradient** to move towards actions that maximize the predicted Q-value while maintaining high entropy to avoid "getting stuck" in dangerous reactor states.

> **Note:** High entropy is critical here—it's the difference between a smooth-running system and a "dangerous reactor state" meltdown. Stay safe out there.

---

# My notes 6:

## subnote 1:

upated -> train.py -> ReplayBuffer, agent.update(), LSTM handling

## subnote 2:

demand.py -> a 24-hour cycle with "solar/wind noise," forcing SAC agent to constantly adjust the control rods without triggering a Xenon spike.

## subnote 3:

dashboard.py -> matplotlib to create a live-updating view of the reactor's "health."

updated -> train.py

updated -> reactor_env.py -> making agent to be more precise



