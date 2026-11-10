cmake_minimum_required(VERSION 3.15...3.31)
project(norman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/src/reactor.cpp cpp/src/bindings.cpp)
target_include_directories(_core PRIVATE cpp/include)

if(MSVC)
  target_compile_options(_core PRIVATE /W4 /O2)
else()
  target_compile_options(_core PRIVATE -Wall -Wextra -O3)
endif()

install(TARGETS _core DESTINATION norman)
