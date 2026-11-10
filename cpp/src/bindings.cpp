#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "norman/reactor.hpp"

namespace py = pybind11;
using norman::Params;
using norman::Reactor;
using norman::Reactivity;

namespace {

py::dict to_dict(const Reactivity& r) {
    py::dict d;
    d["bias"] = r.bias;
    d["rods"] = r.rods;
    d["boron"] = r.boron;
    d["doppler"] = r.doppler;
    d["moderator"] = r.moderator;
    d["xenon"] = r.xenon;
    d["samarium"] = r.samarium;
    return d;
}

}

PYBIND11_MODULE(_core, m) {
    m.attr("STATE_SIZE") = norman::kStateSize;

    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("beta", &Params::beta)
        .def_readwrite("group_fractions", &Params::group_fractions)
        .def_readwrite("group_decay", &Params::group_decay)
        .def_readwrite("generation_time", &Params::generation_time)
        .def_readwrite("neutron_source", &Params::neutron_source)
        .def_readwrite("rated_power", &Params::rated_power)
        .def_readwrite("fuel_power_fraction", &Params::fuel_power_fraction)
        .def_readwrite("t_inlet", &Params::t_inlet)
        .def_readwrite("t_coolant_nominal", &Params::t_coolant_nominal)
        .def_readwrite("t_fuel_nominal", &Params::t_fuel_nominal)
        .def_readwrite("fuel_heat_capacity", &Params::fuel_heat_capacity)
        .def_readwrite("coolant_heat_capacity", &Params::coolant_heat_capacity)
        .def_readwrite("alpha_fuel", &Params::alpha_fuel)
        .def_readwrite("alpha_moderator", &Params::alpha_moderator)
        .def_readwrite("boron_worth", &Params::boron_worth)
        .def_readwrite("rod_bank_worth", &Params::rod_bank_worth)
        .def_readwrite("rod_speed_max", &Params::rod_speed_max)
        .def_readwrite("boron_rate_max", &Params::boron_rate_max)
        .def_readwrite("boron_max", &Params::boron_max)
        .def_readwrite("nominal_flux", &Params::nominal_flux)
        .def_readwrite("gamma_iodine", &Params::gamma_iodine)
        .def_readwrite("gamma_xenon", &Params::gamma_xenon)
        .def_readwrite("gamma_promethium", &Params::gamma_promethium)
        .def_readwrite("lambda_iodine", &Params::lambda_iodine)
        .def_readwrite("lambda_xenon", &Params::lambda_xenon)
        .def_readwrite("lambda_promethium", &Params::lambda_promethium)
        .def_readwrite("sigma_xenon", &Params::sigma_xenon)
        .def_readwrite("sigma_samarium", &Params::sigma_samarium)
        .def_readwrite("nu", &Params::nu)
        .def_readwrite("substep", &Params::substep)
        .def("validate", &Params::validate);

    py::class_<Reactor>(m, "Reactor")
        .def(py::init<const Params&>(), py::arg("params") = Params())
        .def("reset", &Reactor::reset, py::arg("power"), py::arg("rod"), py::arg("boron"))
        .def("set_state", &Reactor::set_state, py::arg("y"), py::arg("rod"), py::arg("boron"),
             py::arg("bias"), py::arg("time") = 0.0)
        .def("advance", &Reactor::advance, py::arg("duration"), py::arg("rod_command"),
             py::arg("boron_command"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("state", &Reactor::state)
        .def_property_readonly("power", &Reactor::power)
        .def_property_readonly("rod", &Reactor::rod)
        .def_property_readonly("boron", &Reactor::boron)
        .def_property_readonly("bias", &Reactor::bias)
        .def_property_readonly("time", &Reactor::time)
        .def_property_readonly("peak_power", &Reactor::peak_power)
        .def_property_readonly("newton_failures", &Reactor::newton_failures)
        .def("reactivity", &Reactor::reactivity)
        .def("reactivity_components", [](const Reactor& r) { return to_dict(r.reactivity_components()); })
        .def("derivatives", [](const Reactor& r, const norman::Vec& y, double rod, double boron) {
            return norman::derivatives(r.params(), y, rod, boron, r.bias());
        })
        .def("copy", [](const Reactor& r) { return Reactor(r); });
}
