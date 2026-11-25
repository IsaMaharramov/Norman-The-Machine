#include "norman/reactor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace norman {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clamp(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

bool solve_linear(Mat a, Vec& b) {
    for (int col = 0; col < kStateSize; ++col) {
        int pivot = col;
        for (int r = col + 1; r < kStateSize; ++r) {
            if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
        }
        if (std::fabs(a[pivot][col]) < 1e-300) return false;
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }
        for (int r = col + 1; r < kStateSize; ++r) {
            const double factor = a[r][col] / a[col][col];
            if (factor == 0.0) continue;
            for (int c = col; c < kStateSize; ++c) a[r][c] -= factor * a[col][c];
            b[r] -= factor * b[col];
        }
    }
    for (int r = kStateSize - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < kStateSize; ++c) s -= a[r][c] * b[c];
        b[r] = s / a[r][r];
    }
    return true;
}

}

double saturation_pressure(double t_celsius) {
    static const double n[10] = {0.11670521452767e4,  -0.72421316703206e6, -0.17073846940092e2,
                                 0.12020824702470e5,  -0.32325550322333e7, 0.14915108613530e2,
                                 -0.48232657361591e4, 0.40511340542057e6,  -0.23855557567849,
                                 0.65017534844798e3};
    const double t = t_celsius + 273.15;
    const double theta = t + n[8] / (t - n[9]);
    const double a = theta * theta + n[0] * theta + n[1];
    const double b = n[2] * theta * theta + n[3] * theta + n[4];
    const double c = n[5] * theta * theta + n[6] * theta + n[7];
    const double x = 2.0 * c / (-b + std::sqrt(b * b - 4.0 * a * c));
    return x * x * x * x;
}

double Params::sg_conductance() const { return rated_power / (t_inlet - t_saturation_full); }

double Params::steam_pressure_full() const { return saturation_pressure(t_saturation_full); }

double Params::tavg_reference(double load) const { return t_inlet + (t_coolant_nominal - t_inlet) * load; }

double inlet_temperature(const Params& p, const Vec& y) { return p.secondary ? y[kColdLeg] : p.t_inlet; }

double steam_fraction(const Params& p, double t_saturation, double load) {
    const double available = p.turbine_valve_max * saturation_pressure(t_saturation) / p.steam_pressure_full();
    return std::min(load, available);
}

namespace {

double steam_fraction_slope(const Params& p, double t_saturation, double load) {
    const double available = p.turbine_valve_max * saturation_pressure(t_saturation) / p.steam_pressure_full();
    if (load <= available) return 0.0;
    const double h = 1e-4;
    const double slope = (saturation_pressure(t_saturation + h) - saturation_pressure(t_saturation - h)) / (2.0 * h);
    return p.turbine_valve_max * slope / p.steam_pressure_full();
}

}

double Params::coolant_flow_capacity() const {
    return rated_power / (2.0 * (t_coolant_nominal - t_inlet));
}

double Params::fuel_to_coolant_conductance() const {
    return fuel_power_fraction * rated_power / (t_fuel_nominal - t_coolant_nominal);
}

double Params::xenon_burn_rate() const { return sigma_xenon * nominal_flux; }

double Params::samarium_burn_rate() const { return sigma_samarium * nominal_flux; }

double Params::xenon_worth_full_power() const {
    const double gamma = gamma_iodine + gamma_xenon;
    const double burn = xenon_burn_rate();
    return gamma / nu * burn / (lambda_xenon + burn);
}

double Params::samarium_worth_full_power() const { return gamma_promethium / nu; }

double Params::xenon_direct_source() const {
    const double gamma = gamma_iodine + gamma_xenon;
    return gamma_xenon / gamma * (lambda_xenon + xenon_burn_rate());
}

double Params::xenon_iodine_source() const {
    const double gamma = gamma_iodine + gamma_xenon;
    return gamma_iodine / gamma * (lambda_xenon + xenon_burn_rate());
}

void Params::validate() const {
    double sum = 0.0;
    for (double f : group_fractions) sum += f;
    if (std::fabs(sum - 1.0) > 1e-6) throw std::invalid_argument("delayed-neutron group fractions must sum to 1");
    if (!(t_fuel_nominal > t_coolant_nominal && t_coolant_nominal > t_inlet))
        throw std::invalid_argument("nominal temperatures must satisfy T_fuel > T_coolant > T_inlet");
    if (beta <= 0 || generation_time <= 0 || neutron_source < 0 || rated_power <= 0 || fuel_heat_capacity <= 0 ||
        coolant_heat_capacity <= 0 || rod_speed_max <= 0 || boron_rate_max <= 0 || substep <= 0 ||
        hot_leg_time <= 0 || cold_leg_time <= 0 || sg_primary_capacity <= 0 || sg_secondary_capacity <= 0 ||
        turbine_valve_max <= 0)
        throw std::invalid_argument("physical parameters must be positive");
    if (!(t_saturation_full < t_inlet))
        throw std::invalid_argument("full-load saturation temperature must be below the cold-leg temperature");
}

double Inputs::rod_at(double t) const { return clamp(rod0 + rod_rate * t, 0.0, 1.0); }

double Inputs::boron_at(double t) const { return clamp(boron0 + boron_rate * t, 0.0, boron_max); }

double rod_worth(const Params& p, double rod) {
    const double z = clamp(rod, 0.0, 1.0);
    const auto& table = p.rod_worth_table;
    if (table.size() >= 2) {
        const double x = z * static_cast<double>(table.size() - 1);
        const size_t k = std::min(static_cast<size_t>(x), table.size() - 2);
        const double w = x - static_cast<double>(k);
        return (1.0 - w) * table[k] + w * table[k + 1];
    }
    return -p.rod_bank_worth * (z - std::sin(2.0 * kPi * z) / (2.0 * kPi));
}

Reactivity reactivity_components(const Params& p, const Vec& y, double rod, double boron, double bias) {
    Reactivity r;
    r.bias = bias;
    r.rods = rod_worth(p, rod);
    r.boron = p.boron_worth * boron;
    r.doppler = p.alpha_fuel * (y[kFuelTemp] - p.t_fuel_nominal);
    r.moderator = p.alpha_moderator * (y[kCoolantTemp] - p.t_coolant_nominal);
    r.xenon = -p.xenon_worth_full_power() * y[kXenon];
    r.samarium = -p.samarium_worth_full_power() * y[kSamarium];
    return r;
}

Vec derivatives(const Params& p, const Vec& y, double rod, double boron, double bias, double load) {
    const double lam = p.generation_time;
    const double n = y[kPower];
    const double rho = reactivity_components(p, y, rod, boron, bias).total();

    Vec dy{};
    double delayed = 0.0;
    for (int i = 0; i < kGroups; ++i) {
        const double c = y[kPrecursor + i];
        delayed += p.beta * p.group_fractions[i] * c;
        dy[kPrecursor + i] = p.group_decay[i] * (n - c);
    }
    dy[kPower] = (rho - p.beta) / lam * n + delayed / lam + p.neutron_source;

    const double ha = p.fuel_to_coolant_conductance();
    const double wc = p.coolant_flow_capacity();
    const double q = p.rated_power * n;
    const double tf = y[kFuelTemp];
    const double tc = y[kCoolantTemp];
    const double t_in = inlet_temperature(p, y);
    dy[kFuelTemp] = (p.fuel_power_fraction * q - ha * (tf - tc)) / p.fuel_heat_capacity;
    dy[kCoolantTemp] =
        ((1.0 - p.fuel_power_fraction) * q + ha * (tf - tc) - 2.0 * wc * (tc - t_in)) /
        p.coolant_heat_capacity;

    dy[kIodine] = p.lambda_iodine * (n - y[kIodine]);
    dy[kXenon] = p.xenon_direct_source() * n + p.xenon_iodine_source() * y[kIodine] -
                 (p.lambda_xenon + p.xenon_burn_rate() * n) * y[kXenon];
    dy[kPromethium] = p.lambda_promethium * (n - y[kPromethium]);
    dy[kSamarium] = p.samarium_burn_rate() * (y[kPromethium] - n * y[kSamarium]);

    if (p.secondary) {
        const double transfer = p.sg_conductance() * (y[kSgPrimary] - y[kSaturation]);
        dy[kHotLeg] = (2.0 * tc - y[kColdLeg] - y[kHotLeg]) / p.hot_leg_time;
        dy[kSgPrimary] = (wc * (y[kHotLeg] - y[kSgPrimary]) - transfer) / p.sg_primary_capacity;
        dy[kColdLeg] = (y[kSgPrimary] - y[kColdLeg]) / p.cold_leg_time;
        dy[kSaturation] =
            (transfer - p.rated_power * steam_fraction(p, y[kSaturation], load)) / p.sg_secondary_capacity;
    }
    return dy;
}

Mat jacobian(const Params& p, const Vec& y, double rod, double boron, double bias, double load) {
    const double lam = p.generation_time;
    const double n = y[kPower];
    const double rho = reactivity_components(p, y, rod, boron, bias).total();
    Mat J{};

    J[kPower][kPower] = (rho - p.beta) / lam;
    for (int i = 0; i < kGroups; ++i) {
        J[kPower][kPrecursor + i] = p.beta * p.group_fractions[i] / lam;
        J[kPrecursor + i][kPower] = p.group_decay[i];
        J[kPrecursor + i][kPrecursor + i] = -p.group_decay[i];
    }
    J[kPower][kFuelTemp] = p.alpha_fuel * n / lam;
    J[kPower][kCoolantTemp] = p.alpha_moderator * n / lam;
    J[kPower][kXenon] = -p.xenon_worth_full_power() * n / lam;
    J[kPower][kSamarium] = -p.samarium_worth_full_power() * n / lam;

    const double ha = p.fuel_to_coolant_conductance();
    const double wc = p.coolant_flow_capacity();
    const double cf = p.fuel_heat_capacity;
    const double cc = p.coolant_heat_capacity;
    J[kFuelTemp][kPower] = p.fuel_power_fraction * p.rated_power / cf;
    J[kFuelTemp][kFuelTemp] = -ha / cf;
    J[kFuelTemp][kCoolantTemp] = ha / cf;
    J[kCoolantTemp][kPower] = (1.0 - p.fuel_power_fraction) * p.rated_power / cc;
    J[kCoolantTemp][kFuelTemp] = ha / cc;
    J[kCoolantTemp][kCoolantTemp] = -(ha + 2.0 * wc) / cc;

    J[kIodine][kPower] = p.lambda_iodine;
    J[kIodine][kIodine] = -p.lambda_iodine;
    J[kXenon][kPower] = p.xenon_direct_source() - p.xenon_burn_rate() * y[kXenon];
    J[kXenon][kIodine] = p.xenon_iodine_source();
    J[kXenon][kXenon] = -(p.lambda_xenon + p.xenon_burn_rate() * n);
    J[kPromethium][kPower] = p.lambda_promethium;
    J[kPromethium][kPromethium] = -p.lambda_promethium;
    J[kSamarium][kPower] = -p.samarium_burn_rate() * y[kSamarium];
    J[kSamarium][kPromethium] = p.samarium_burn_rate();
    J[kSamarium][kSamarium] = -p.samarium_burn_rate() * n;

    if (p.secondary) {
        const double ua = p.sg_conductance();
        const double cp = p.sg_primary_capacity;
        const double cs = p.sg_secondary_capacity;
        J[kCoolantTemp][kColdLeg] = 2.0 * wc / cc;
        J[kHotLeg][kCoolantTemp] = 2.0 / p.hot_leg_time;
        J[kHotLeg][kColdLeg] = -1.0 / p.hot_leg_time;
        J[kHotLeg][kHotLeg] = -1.0 / p.hot_leg_time;
        J[kSgPrimary][kHotLeg] = wc / cp;
        J[kSgPrimary][kSgPrimary] = -(wc + ua) / cp;
        J[kSgPrimary][kSaturation] = ua / cp;
        J[kColdLeg][kSgPrimary] = 1.0 / p.cold_leg_time;
        J[kColdLeg][kColdLeg] = -1.0 / p.cold_leg_time;
        J[kSaturation][kSgPrimary] = ua / cs;
        J[kSaturation][kSaturation] = -(ua + p.rated_power * steam_fraction_slope(p, y[kSaturation], load)) / cs;
    }
    return J;
}

Vec equilibrium_state(const Params& p, double power) {
    if (!(power > 0.0)) throw std::invalid_argument("equilibrium requires a positive power level");
    Vec y{};
    y[kPower] = power;
    for (int i = 0; i < kGroups; ++i) y[kPrecursor + i] = power;
    const double tc = p.tavg_reference(power);
    const double t_cold = tc - p.rated_power * power / (2.0 * p.coolant_flow_capacity());
    y[kCoolantTemp] = tc;
    y[kHotLeg] = 2.0 * tc - t_cold;
    y[kSgPrimary] = t_cold;
    y[kColdLeg] = t_cold;
    y[kSaturation] = t_cold - p.rated_power * power / p.sg_conductance();
    y[kFuelTemp] = tc + p.fuel_power_fraction * p.rated_power * power / p.fuel_to_coolant_conductance();
    y[kIodine] = power;
    const double burn = p.xenon_burn_rate();
    y[kXenon] = power * (p.lambda_xenon + burn) / (p.lambda_xenon + burn * power);
    y[kPromethium] = power;
    y[kSamarium] = 1.0;
    return y;
}

Reactor::Reactor(const Params& params) : params_(params) {
    params_.validate();
    reset(1.0, 0.25, 0.0);
}

void Reactor::reset(double power, double rod, double boron) {
    y_ = equilibrium_state(params_, power);
    rod_ = clamp(rod, 0.0, 1.0);
    boron_ = clamp(boron, 0.0, params_.boron_max);
    bias_ = -norman::reactivity_components(params_, y_, rod_, boron_, 0.0).total() -
            params_.generation_time * params_.neutron_source / power;
    time_ = 0.0;
    peak_power_ = power;
    load_ = power;
}

double Reactor::steam_power() const {
    if (!params_.secondary) return y_[kPower];
    return steam_fraction(params_, y_[kSaturation], load_);
}

void Reactor::set_rod_worth_table(const std::vector<double>& table) {
    if (table.size() == 1) throw std::invalid_argument("a rod worth table needs at least two points");
    params_.rod_worth_table = table;
}

void Reactor::set_state(const Vec& y, double rod, double boron, double bias, double time) {
    y_ = y;
    rod_ = rod;
    boron_ = boron;
    bias_ = bias;
    time_ = time;
    peak_power_ = y[kPower];
}

Reactivity Reactor::reactivity_components() const {
    return norman::reactivity_components(params_, y_, rod_, boron_, bias_);
}

bool Reactor::newton(Vec& Y, const Vec& base, double t, double hg, const Inputs& in) const {
    const double rod = in.rod_at(t);
    const double boron = in.boron_at(t);
    const double load = in.load_at(t);
    for (int iter = 0; iter < 30; ++iter) {
        const Vec f = derivatives(params_, Y, rod, boron, bias_, load);
        const Mat J = jacobian(params_, Y, rod, boron, bias_, load);
        Mat A{};
        Vec rhs{};
        for (int r = 0; r < kStateSize; ++r) {
            for (int c = 0; c < kStateSize; ++c) A[r][c] = -hg * J[r][c];
            A[r][r] += 1.0;
            rhs[r] = -(Y[r] - base[r] - hg * f[r]);
        }
        if (!solve_linear(A, rhs)) return false;
        bool converged = true;
        for (int r = 0; r < kStateSize; ++r) {
            Y[r] += rhs[r];
            if (!std::isfinite(Y[r])) return false;
            if (std::fabs(rhs[r]) > 1e-18 + 1e-11 * std::fabs(Y[r])) converged = false;
        }
        if (converged) return true;
    }
    return false;
}

bool Reactor::sdirk_step(Vec& y, double t, double h, const Inputs& in) const {
    const double g = 1.0 - std::sqrt(2.0) / 2.0;
    Vec y1 = y;
    if (!newton(y1, y, t + g * h, g * h, in)) return false;
    const Vec k1 =
        derivatives(params_, y1, in.rod_at(t + g * h), in.boron_at(t + g * h), bias_, in.load_at(t + g * h));
    Vec base2{};
    for (int r = 0; r < kStateSize; ++r) base2[r] = y[r] + h * (1.0 - g) * k1[r];
    Vec y2 = y1;
    if (!newton(y2, base2, t + h, g * h, in)) return false;
    y = y2;
    return true;
}

void Reactor::step_adaptive(Vec& y, double t, double h, const Inputs& in, int depth) {
    Vec trial = y;
    if (sdirk_step(trial, t, h, in)) {
        y = trial;
        return;
    }
    if (depth >= 12) throw std::runtime_error("reactor integration failed to converge");
    ++newton_failures_;
    step_adaptive(y, t, 0.5 * h, in, depth + 1);
    step_adaptive(y, t + 0.5 * h, 0.5 * h, in, depth + 1);
}

void Reactor::advance(double duration, double rod_command, double boron_command, double load) {
    if (!(duration > 0.0)) throw std::invalid_argument("duration must be positive");
    Inputs in;
    in.rod0 = rod_;
    in.rod_rate = -clamp(rod_command, -1.0, 1.0) * params_.rod_speed_max;
    in.boron0 = boron_;
    in.boron_rate = -clamp(boron_command, -1.0, 1.0) * params_.boron_rate_max;
    in.boron_max = params_.boron_max;
    in.load0 = load_;
    in.load_rate = load < 0.0 ? 0.0 : (clamp(load, 0.0, 1.2) - load_) / duration;

    const int steps = std::max(1, static_cast<int>(std::ceil(duration / params_.substep - 1e-9)));
    const double h = duration / steps;
    double peak = y_[kPower];
    for (int k = 0; k < steps; ++k) {
        step_adaptive(y_, k * h, h, in, 0);
        peak = std::max(peak, y_[kPower]);
    }
    rod_ = in.rod_at(duration);
    boron_ = in.boron_at(duration);
    load_ = in.load_at(duration);
    time_ += duration;
    peak_power_ = peak;
}

}
