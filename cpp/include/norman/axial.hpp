#pragma once

#include <vector>

#include "norman/reactor.hpp"

namespace norman {

struct AxialParams {
    int nodes = 40;
    double height = 366.0;
    double extrapolation = 8.0;
    double migration_area = 50.0;
    double design_axial_offset = -0.03;
    int max_shape_iterations = 200;
    double shape_tolerance = 1e-11;
    double poison_substep = 300.0;

    void validate() const;
};

double smallest_eigenvalue(const std::vector<double>& diag, const std::vector<double>& off);

class AxialCore {
public:
    AxialCore(const Params& params, const AxialParams& axial);

    void initialize(double power, double rod, double boron);
    double update(double power, double rod, double boron, double coolant_power = -1.0, double fuel_power = -1.0,
                  double inlet = -1.0);
    void advance_poisons(double dt, double mean_power);
    double solve_eigen(const std::vector<double>& local_reactivity, std::vector<double>& shape) const;

    const Params& params() const { return params_; }
    const AxialParams& axial_params() const { return axial_; }
    int nodes() const { return axial_.nodes; }
    double node_height() const { return h_; }
    double k_infinity_reference() const { return k_inf0_; }

    const std::vector<double>& shape() const { return shape_; }
    const std::vector<double>& fuel_temperature() const { return t_fuel_; }
    const std::vector<double>& coolant_temperature() const { return t_coolant_; }
    const std::vector<double>& iodine() const { return iodine_; }
    const std::vector<double>& xenon() const { return xenon_; }
    const std::vector<double>& promethium() const { return promethium_; }
    const std::vector<double>& samarium() const { return samarium_; }
    const std::vector<double>& local_reactivity() const { return rho_local_; }
    std::vector<double> rod_coverage(double rod) const;
    std::vector<double> rod_worth_curve(int points, double boron) const;

    double reactivity() const { return reactivity_; }
    double k_eff() const { return 1.0 / (1.0 - reactivity_); }
    double bias() const { return bias_; }
    double tilt() const { return tilt_; }
    double axial_offset() const;
    double xenon_axial_offset() const;
    double peaking() const;
    double inlet() const { return inlet_; }

    void set_poisons(const std::vector<double>& iodine, const std::vector<double>& xenon,
                     const std::vector<double>& promethium, const std::vector<double>& samarium);

private:
    void build_leakage();
    void quasi_steady_temperatures(double coolant_power, double fuel_power);
    void equilibrium_poisons(double power);
    void local_reactivities(double rod, double boron);
    std::vector<double> compute_local(double rod, double boron) const;
    double iterate_shape(double power, double rod, double boron, bool equilibrium, bool hold_critical,
                         double coolant_power, double fuel_power);
    double half_balance(const std::vector<double>& values) const;

    Params params_;
    AxialParams axial_;
    double h_ = 0.0;
    double k_inf0_ = 1.0;
    std::vector<double> a_diag_, a_off_;
    std::vector<double> shape_, t_fuel_, t_coolant_;
    std::vector<double> iodine_, xenon_, promethium_, samarium_, rho_local_;
    double bias_ = 0.0;
    double tilt_ = 0.0;
    double reactivity_ = 0.0;
    double inlet_ = 0.0;
};

}
