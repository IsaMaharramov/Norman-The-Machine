#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "norman/axial.hpp"
#include "norman/reactor.hpp"

namespace py = pybind11;
using norman::AxialCore;
using norman::AxialParams;
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
        .def_readwrite("rod_worth_table", &Params::rod_worth_table)
        .def_readwrite("secondary", &Params::secondary)
        .def_readwrite("hot_leg_time", &Params::hot_leg_time)
        .def_readwrite("cold_leg_time", &Params::cold_leg_time)
        .def_readwrite("sg_primary_capacity", &Params::sg_primary_capacity)
        .def_readwrite("sg_secondary_capacity", &Params::sg_secondary_capacity)
        .def_readwrite("t_saturation_full", &Params::t_saturation_full)
        .def_readwrite("turbine_valve_max", &Params::turbine_valve_max)
        .def("validate", &Params::validate);

    py::class_<Reactor>(m, "Reactor")
        .def(py::init<const Params&>(), py::arg("params") = Params())
        .def("reset", &Reactor::reset, py::arg("power"), py::arg("rod"), py::arg("boron"))
        .def("set_state", &Reactor::set_state, py::arg("y"), py::arg("rod"), py::arg("boron"),
             py::arg("bias"), py::arg("time") = 0.0)
        .def("advance", &Reactor::advance, py::arg("duration"), py::arg("rod_command"),
             py::arg("boron_command"), py::arg("load") = -1.0, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("load", &Reactor::load)
        .def_property_readonly("steam_power", &Reactor::steam_power)
        .def_property_readonly("inlet_temperature", &Reactor::inlet)
        .def_property_readonly("state", &Reactor::state)
        .def_property_readonly("power", &Reactor::power)
        .def_property_readonly("rod", &Reactor::rod)
        .def_property_readonly("boron", &Reactor::boron)
        .def_property_readonly("bias", &Reactor::bias)
        .def_property_readonly("time", &Reactor::time)
        .def_property_readonly("peak_power", &Reactor::peak_power)
        .def_property_readonly("newton_failures", &Reactor::newton_failures)
        .def("set_rod_worth_table", &Reactor::set_rod_worth_table, py::arg("table"))
        .def("reactivity", &Reactor::reactivity)
        .def("reactivity_components", [](const Reactor& r) { return to_dict(r.reactivity_components()); })
        .def("derivatives", [](const Reactor& r, const norman::Vec& y, double rod, double boron) {
            return norman::derivatives(r.params(), y, rod, boron, r.bias(), r.load());
        })
        .def("copy", [](const Reactor& r) { return Reactor(r); });

    py::class_<AxialParams>(m, "AxialParams")
        .def(py::init<>())
        .def_readwrite("nodes", &AxialParams::nodes)
        .def_readwrite("height", &AxialParams::height)
        .def_readwrite("extrapolation", &AxialParams::extrapolation)
        .def_readwrite("migration_area", &AxialParams::migration_area)
        .def_readwrite("design_axial_offset", &AxialParams::design_axial_offset)
        .def_readwrite("max_shape_iterations", &AxialParams::max_shape_iterations)
        .def_readwrite("shape_tolerance", &AxialParams::shape_tolerance)
        .def_readwrite("poison_substep", &AxialParams::poison_substep)
        .def("validate", &AxialParams::validate);

    m.def("saturation_pressure", &norman::saturation_pressure, py::arg("t_celsius"));
    m.def("smallest_eigenvalue", &norman::smallest_eigenvalue, py::arg("diag"), py::arg("off"));

    py::class_<AxialCore>(m, "AxialCore")
        .def(py::init<const Params&, const AxialParams&>(), py::arg("params"), py::arg("axial"))
        .def("initialize", &AxialCore::initialize, py::arg("power"), py::arg("rod"), py::arg("boron"))
        .def("update", &AxialCore::update, py::arg("power"), py::arg("rod"), py::arg("boron"),
             py::arg("coolant_power") = -1.0, py::arg("fuel_power") = -1.0, py::arg("inlet") = -1.0)
        .def("advance_poisons", &AxialCore::advance_poisons, py::arg("dt"), py::arg("mean_power"))
        .def("solve_eigen", [](const AxialCore& c, const std::vector<double>& rho) {
            std::vector<double> shape;
            const double lambda = c.solve_eigen(rho, shape);
            return py::make_tuple(lambda, shape);
        })
        .def("rod_coverage", &AxialCore::rod_coverage)
        .def("rod_worth_curve", &AxialCore::rod_worth_curve, py::arg("points"), py::arg("boron"))
        .def("set_poisons", &AxialCore::set_poisons)
        .def_property_readonly("nodes", &AxialCore::nodes)
        .def_property_readonly("node_height", &AxialCore::node_height)
        .def_property_readonly("k_infinity_reference", &AxialCore::k_infinity_reference)
        .def_property_readonly("shape", &AxialCore::shape)
        .def_property_readonly("fuel_temperature", &AxialCore::fuel_temperature)
        .def_property_readonly("coolant_temperature", &AxialCore::coolant_temperature)
        .def_property_readonly("iodine", &AxialCore::iodine)
        .def_property_readonly("xenon", &AxialCore::xenon)
        .def_property_readonly("promethium", &AxialCore::promethium)
        .def_property_readonly("samarium", &AxialCore::samarium)
        .def_property_readonly("local_reactivity", &AxialCore::local_reactivity)
        .def_property_readonly("reactivity", &AxialCore::reactivity)
        .def_property_readonly("k_eff", &AxialCore::k_eff)
        .def_property_readonly("bias", &AxialCore::bias)
        .def_property_readonly("tilt", &AxialCore::tilt)
        .def_property_readonly("axial_offset", &AxialCore::axial_offset)
        .def_property_readonly("xenon_axial_offset", &AxialCore::xenon_axial_offset)
        .def_property_readonly("peaking", &AxialCore::peaking)
        .def_property_readonly("inlet", &AxialCore::inlet)
        .def("copy", [](const AxialCore& c) { return AxialCore(c); });
}
