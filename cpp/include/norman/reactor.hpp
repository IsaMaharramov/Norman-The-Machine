#pragma once

#include <array>

namespace norman {

constexpr int kStateSize = 13;
constexpr int kGroups = 6;

enum Index : int {
    kPower = 0,
    kPrecursor = 1,
    kFuelTemp = 7,
    kCoolantTemp = 8,
    kIodine = 9,
    kXenon = 10,
    kPromethium = 11,
    kSamarium = 12,
};

using Vec = std::array<double, kStateSize>;
using Mat = std::array<Vec, kStateSize>;

struct Params {
    double beta = 0.0058;
    std::array<double, kGroups> group_fractions{0.033, 0.219, 0.196, 0.395, 0.115, 0.042};
    std::array<double, kGroups> group_decay{0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01};
    double generation_time = 2.0e-5;
    double neutron_source = 5.0e-7;

    double rated_power = 3.0e9;
    double fuel_power_fraction = 0.974;
    double t_inlet = 292.0;
    double t_coolant_nominal = 308.0;
    double t_fuel_nominal = 650.0;
    double fuel_heat_capacity = 4.27e7;
    double coolant_heat_capacity = 1.2e8;

    double alpha_fuel = -2.5e-5;
    double alpha_moderator = -25.0e-5;
    double boron_worth = -8.0e-5;

    double rod_bank_worth = 2500.0e-5;
    double rod_speed_max = 1.0 / 180.0;
    double boron_rate_max = 3.0 / 60.0;
    double boron_max = 2000.0;

    double nominal_flux = 3.0e13;
    double gamma_iodine = 0.0639;
    double gamma_xenon = 0.00237;
    double gamma_promethium = 0.0113;
    double lambda_iodine = 2.87e-5;
    double lambda_xenon = 2.09e-5;
    double lambda_promethium = 3.63e-6;
    double sigma_xenon = 2.65e-18;
    double sigma_samarium = 4.08e-20;
    double nu = 2.43;

    double substep = 0.25;

    double coolant_flow_capacity() const;
    double fuel_to_coolant_conductance() const;
    double xenon_burn_rate() const;
    double samarium_burn_rate() const;
    double xenon_worth_full_power() const;
    double samarium_worth_full_power() const;
    double xenon_direct_source() const;
    double xenon_iodine_source() const;
    void validate() const;
};

struct Reactivity {
    double bias = 0.0;
    double rods = 0.0;
    double boron = 0.0;
    double doppler = 0.0;
    double moderator = 0.0;
    double xenon = 0.0;
    double samarium = 0.0;
    double total() const { return bias + rods + boron + doppler + moderator + xenon + samarium; }
};

struct Inputs {
    double rod0 = 0.0;
    double rod_rate = 0.0;
    double boron0 = 0.0;
    double boron_rate = 0.0;
    double boron_max = 0.0;
    double rod_at(double t) const;
    double boron_at(double t) const;
};

double rod_worth(const Params& p, double rod);
Reactivity reactivity_components(const Params& p, const Vec& y, double rod, double boron, double bias);
Vec derivatives(const Params& p, const Vec& y, double rod, double boron, double bias);
Mat jacobian(const Params& p, const Vec& y, double rod, double boron, double bias);
Vec equilibrium_state(const Params& p, double power);

class Reactor {
public:
    explicit Reactor(const Params& params = Params());

    void reset(double power, double rod, double boron);
    void set_state(const Vec& y, double rod, double boron, double bias, double time);
    void advance(double duration, double rod_command, double boron_command);

    const Params& params() const { return params_; }
    const Vec& state() const { return y_; }
    double power() const { return y_[kPower]; }
    double rod() const { return rod_; }
    double boron() const { return boron_; }
    double bias() const { return bias_; }
    double time() const { return time_; }
    double peak_power() const { return peak_power_; }
    long newton_failures() const { return newton_failures_; }

    Reactivity reactivity_components() const;
    double reactivity() const { return reactivity_components().total(); }

private:
    bool sdirk_step(Vec& y, double t, double h, const Inputs& in) const;
    bool newton(Vec& Y, const Vec& base, double t, double hg, const Inputs& in) const;
    void step_adaptive(Vec& y, double t, double h, const Inputs& in, int depth);

    Params params_;
    Vec y_{};
    double rod_ = 0.25;
    double boron_ = 0.0;
    double bias_ = 0.0;
    double time_ = 0.0;
    double peak_power_ = 1.0;
    long newton_failures_ = 0;
};

}
