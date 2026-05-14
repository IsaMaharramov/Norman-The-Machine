# What is my notes?
I note ~ everything I do

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


# My notes 2:

We will implement the physics using a **4th-order Runge-Kutta (RK4) integrator** instead of a simple Euler method. Why? Because nuclear reactor kinetics are "stiff" differential equations;

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

# My notes 4:

