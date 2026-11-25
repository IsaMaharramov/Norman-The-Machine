#include "norman/axial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace norman {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clamp(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

int count_below(const std::vector<double>& d, const std::vector<double>& e, double x) {
    int count = 0;
    double q = d[0] - x;
    if (q < 0.0) ++count;
    for (size_t i = 1; i < d.size(); ++i) {
        if (q == 0.0) q = 1e-300;
        q = d[i] - x - e[i - 1] * e[i - 1] / q;
        if (q < 0.0) ++count;
    }
    return count;
}

void thomas(const std::vector<double>& diag, const std::vector<double>& off, double shift,
            std::vector<double>& rhs) {
    const size_t n = diag.size();
    std::vector<double> c(n, 0.0);
    double denom = diag[0] - shift;
    c[0] = n > 1 ? off[0] / denom : 0.0;
    rhs[0] /= denom;
    for (size_t i = 1; i < n; ++i) {
        denom = diag[i] - shift - off[i - 1] * c[i - 1];
        if (i + 1 < n) c[i] = off[i] / denom;
        rhs[i] = (rhs[i] - off[i - 1] * rhs[i - 1]) / denom;
    }
    for (size_t i = n - 1; i-- > 0;) rhs[i] -= c[i] * rhs[i + 1];
}

}

void AxialParams::validate() const {
    if (nodes < 4 || nodes % 2 != 0) throw std::invalid_argument("nodes must be an even number >= 4");
    if (height <= 0 || extrapolation < 0 || migration_area <= 0 || poison_substep <= 0)
        throw std::invalid_argument("axial geometry parameters must be positive");
    if (max_shape_iterations < 1 || shape_tolerance <= 0)
        throw std::invalid_argument("shape iteration settings must be positive");
}

double smallest_eigenvalue(const std::vector<double>& d, const std::vector<double>& e) {
    const size_t n = d.size();
    double lo = d[0], hi = d[0];
    for (size_t i = 0; i < n; ++i) {
        const double r = (i > 0 ? std::fabs(e[i - 1]) : 0.0) + (i + 1 < n ? std::fabs(e[i]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
    }
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        if (count_below(d, e, mid) >= 1) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

AxialCore::AxialCore(const Params& params, const AxialParams& axial) : params_(params), axial_(axial) {
    params_.validate();
    axial_.validate();
    const int n = axial_.nodes;
    h_ = axial_.height / n;
    build_leakage();
    k_inf0_ = smallest_eigenvalue(a_diag_, a_off_);
    shape_.assign(n, 1.0);
    for (int i = 0; i < n; ++i) shape_[i] = std::sin(kPi * (i + 0.5) / n);
    double mean = 0.0;
    for (double s : shape_) mean += s;
    mean /= n;
    for (double& s : shape_) s /= mean;
    t_fuel_.assign(n, params_.t_fuel_nominal);
    t_coolant_.assign(n, params_.t_coolant_nominal);
    iodine_.assign(n, 1.0);
    xenon_.assign(n, 1.0);
    promethium_.assign(n, 1.0);
    samarium_.assign(n, 1.0);
    rho_local_.assign(n, 0.0);
    inlet_ = params_.t_inlet;
}

void AxialCore::build_leakage() {
    const int n = axial_.nodes;
    const double m2 = axial_.migration_area;
    const double inner = m2 / (h_ * h_);
    const double edge = m2 / (h_ * (0.5 * h_ + axial_.extrapolation));
    a_diag_.assign(n, 0.0);
    a_off_.assign(n - 1, -inner);
    for (int i = 0; i < n; ++i) {
        const double left = i == 0 ? edge : inner;
        const double right = i == n - 1 ? edge : inner;
        a_diag_[i] = 1.0 + left + right;
    }
}

double AxialCore::solve_eigen(const std::vector<double>& rho, std::vector<double>& shape) const {
    const int n = axial_.nodes;
    std::vector<double> f(n), d(n), e(n - 1);
    for (int i = 0; i < n; ++i) {
        f[i] = k_inf0_ / (1.0 - rho[i]);
        d[i] = a_diag_[i] / f[i];
    }
    for (int i = 0; i < n - 1; ++i) e[i] = a_off_[i] / std::sqrt(f[i] * f[i + 1]);

    const double lambda = smallest_eigenvalue(d, e);
    const double shift = lambda - 1e-9 * std::max(1.0, std::fabs(lambda));
    std::vector<double> v(n, 1.0);
    for (int it = 0; it < 3; ++it) {
        thomas(d, e, shift, v);
        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (double& x : v) x /= norm;
    }
    shape.assign(n, 0.0);
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
        shape[i] = v[i] / std::sqrt(f[i]);
        mean += shape[i];
    }
    mean /= n;
    for (double& s : shape) s /= mean;
    return lambda;
}

std::vector<double> AxialCore::rod_coverage(double rod) const {
    const int n = axial_.nodes;
    const double tip = axial_.height * (1.0 - clamp(rod, 0.0, 1.0));
    std::vector<double> cov(n);
    for (int i = 0; i < n; ++i) cov[i] = clamp(((i + 1) * h_ - tip) / h_, 0.0, 1.0);
    return cov;
}

void AxialCore::quasi_steady_temperatures(double coolant_power, double fuel_power) {
    const int n = axial_.nodes;
    const double flow = params_.coolant_flow_capacity();
    const double conductance = params_.fuel_to_coolant_conductance() / n;
    double t_in = inlet_;
    for (int i = 0; i < n; ++i) {
        const double node = shape_[i] * params_.rated_power / n;
        const double t_out = t_in + coolant_power * node / flow;
        t_coolant_[i] = 0.5 * (t_in + t_out);
        t_fuel_[i] = t_coolant_[i] + params_.fuel_power_fraction * fuel_power * node / conductance;
        t_in = t_out;
    }
}

void AxialCore::equilibrium_poisons(double power) {
    const double burn = params_.xenon_burn_rate();
    const double lx = params_.lambda_xenon;
    for (int i = 0; i < axial_.nodes; ++i) {
        const double u = power * shape_[i];
        iodine_[i] = u;
        xenon_[i] = u * (lx + burn) / (lx + burn * u);
        promethium_[i] = u;
        if (u > 0.0) samarium_[i] = 1.0;
    }
}

void AxialCore::local_reactivities(double rod, double boron) { rho_local_ = compute_local(rod, boron); }

std::vector<double> AxialCore::rod_worth_curve(int points, double boron) const {
    if (points < 2) throw std::invalid_argument("a rod worth curve needs at least two points");
    std::vector<double> shape;
    std::vector<double> curve(points);
    double reference = 0.0;
    for (int k = 0; k < points; ++k) {
        const double z = static_cast<double>(k) / (points - 1);
        const double rho = 1.0 - solve_eigen(compute_local(z, boron), shape);
        if (k == 0) reference = rho;
        curve[k] = rho - reference;
    }
    return curve;
}

std::vector<double> AxialCore::compute_local(double rod, double boron) const {
    const int n = axial_.nodes;
    const std::vector<double> cov = rod_coverage(rod);
    const double wx = params_.xenon_worth_full_power();
    const double wsm = params_.samarium_worth_full_power();
    std::vector<double> rho_local(n);
    for (int i = 0; i < n; ++i) {
        const double z = (i + 0.5) / n;
        rho_local[i] = bias_ + tilt_ * (z - 0.5) - params_.rod_bank_worth * cov[i] +
                        params_.boron_worth * boron +
                        params_.alpha_fuel * (t_fuel_[i] - params_.t_fuel_nominal) +
                        params_.alpha_moderator * (t_coolant_[i] - params_.t_coolant_nominal) -
                        wx * xenon_[i] - wsm * samarium_[i];
    }
    return rho_local;
}

double AxialCore::iterate_shape(double power, double rod, double boron, bool equilibrium, bool hold_critical,
                                double coolant_power, double fuel_power) {
    std::vector<double> next;
    double rho = 0.0;
    for (int it = 0; it < axial_.max_shape_iterations; ++it) {
        quasi_steady_temperatures(coolant_power, fuel_power);
        if (equilibrium) equilibrium_poisons(power);
        local_reactivities(rod, boron);
        rho = 1.0 - solve_eigen(rho_local_, next);
        double change = 0.0;
        for (int i = 0; i < axial_.nodes; ++i) {
            change = std::max(change, std::fabs(next[i] - shape_[i]));
            shape_[i] = 0.3 * shape_[i] + 0.7 * next[i];
        }
        if (hold_critical) bias_ -= rho;
        if (change < axial_.shape_tolerance && (!hold_critical || std::fabs(rho) < axial_.shape_tolerance)) break;
    }
    quasi_steady_temperatures(coolant_power, fuel_power);
    if (equilibrium) equilibrium_poisons(power);
    local_reactivities(rod, boron);
    rho = 1.0 - solve_eigen(rho_local_, next);
    shape_ = next;
    return rho;
}

void AxialCore::initialize(double power, double rod, double boron) {
    if (!(power > 0.0)) throw std::invalid_argument("initial power must be positive");
    inlet_ = params_.t_inlet;
    bias_ = 0.0;
    double t0 = 0.0, t1 = 0.01;
    tilt_ = t0;
    iterate_shape(1.0, rod, boron, true, true, 1.0, 1.0);
    double f0 = axial_offset() - axial_.design_axial_offset;
    tilt_ = t1;
    iterate_shape(1.0, rod, boron, true, true, 1.0, 1.0);
    double f1 = axial_offset() - axial_.design_axial_offset;
    for (int it = 0; it < 40 && std::fabs(f1) > 1e-10; ++it) {
        if (f1 == f0) break;
        const double t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
        t0 = t1;
        f0 = f1;
        t1 = t2;
        tilt_ = t1;
        iterate_shape(1.0, rod, boron, true, true, 1.0, 1.0);
        f1 = axial_offset() - axial_.design_axial_offset;
    }
    reactivity_ = iterate_shape(power, rod, boron, true, true, power, power);
}

double AxialCore::update(double power, double rod, double boron, double coolant_power, double fuel_power,
                         double inlet) {
    inlet_ = inlet < 0.0 ? params_.t_inlet : inlet;
    if (coolant_power < 0.0) coolant_power = power;
    if (fuel_power < 0.0) fuel_power = power;
    reactivity_ = iterate_shape(power, rod, boron, false, false, coolant_power, fuel_power);
    return reactivity_;
}

void AxialCore::advance_poisons(double dt, double mean_power) {
    if (!(dt > 0.0)) throw std::invalid_argument("dt must be positive");
    const int steps = std::max(1, static_cast<int>(std::ceil(dt / axial_.poison_substep - 1e-9)));
    const double hs = dt / steps;
    const double li = params_.lambda_iodine, lx = params_.lambda_xenon, lp = params_.lambda_promethium;
    const double burn = params_.xenon_burn_rate(), burn_sm = params_.samarium_burn_rate();
    const double a = params_.xenon_direct_source(), b = params_.xenon_iodine_source();
    const double ei = std::exp(-li * hs), ep = std::exp(-lp * hs);
    for (int i = 0; i < axial_.nodes; ++i) {
        const double u = mean_power * shape_[i];
        for (int k = 0; k < steps; ++k) {
            const double i_new = u + (iodine_[i] - u) * ei;
            const double r = lx + burn * u;
            xenon_[i] = (xenon_[i] * (1.0 - 0.5 * hs * r) + hs * (a * u + b * 0.5 * (iodine_[i] + i_new))) /
                        (1.0 + 0.5 * hs * r);
            iodine_[i] = i_new;
            const double p_new = u + (promethium_[i] - u) * ep;
            const double q = burn_sm * u;
            samarium_[i] = (samarium_[i] * (1.0 - 0.5 * hs * q) + hs * burn_sm * 0.5 * (promethium_[i] + p_new)) /
                           (1.0 + 0.5 * hs * q);
            promethium_[i] = p_new;
        }
    }
}

void AxialCore::set_poisons(const std::vector<double>& iodine, const std::vector<double>& xenon,
                            const std::vector<double>& promethium, const std::vector<double>& samarium) {
    const size_t n = static_cast<size_t>(axial_.nodes);
    if (iodine.size() != n || xenon.size() != n || promethium.size() != n || samarium.size() != n)
        throw std::invalid_argument("poison profiles must have one value per node");
    iodine_ = iodine;
    xenon_ = xenon;
    promethium_ = promethium;
    samarium_ = samarium;
}

double AxialCore::half_balance(const std::vector<double>& values) const {
    const int half = axial_.nodes / 2;
    double bottom = 0.0, top = 0.0;
    for (int i = 0; i < half; ++i) bottom += values[i];
    for (int i = half; i < axial_.nodes; ++i) top += values[i];
    return (top - bottom) / (top + bottom);
}

double AxialCore::axial_offset() const { return half_balance(shape_); }

double AxialCore::xenon_axial_offset() const { return half_balance(xenon_); }

double AxialCore::peaking() const { return *std::max_element(shape_.begin(), shape_.end()); }

}
