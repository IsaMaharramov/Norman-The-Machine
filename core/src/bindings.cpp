#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ReactorDynamics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(norman_core, m)
{
    m.doc() = "Norman_The_Machine: High-performance nuclear physics core";

    py::class_<ReactorState>(m, "ReactorState")
        .def_readonly("neutron_flux", &ReactorState::neutron_flux)
        .def_readonly("iodine_conc", &ReactorState::iodine_conc)
        .def_readonly("xenon_conc", &ReactorState::xenon_conc)
        .def_readonly("power_level", &ReactorState::power_level);

    py::class_<ReactorEngine>(m, "ReactorEngine")
        .def(py::init<double>(), py::arg("dt") = 1.0)
        .def("step", &ReactorEngine::step)
        .def("get_state", &ReactorEngine::get_state)
        .def("reset", &ReactorEngine::reset);
}