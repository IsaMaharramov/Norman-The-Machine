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