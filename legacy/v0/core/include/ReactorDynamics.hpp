#ifndef REACTOR_DYNAMICS_HPP
#define REACTOR_DYNAMICS_HPP

#include <vector>

struct ReactorState
{
    double neutron_flux; // Φ(t)
    double iodine_conc;  // I(t)
    double xenon_conc;   // X(t)
    double power_level;  // 0.0 to 1.0 (normalized)
};

class ReactorEngine
{
public:
    ReactorEngine(double dt = 1.0);

    void step(double control_rod_pos);

    ReactorState get_state() const { return state; }
    void reset();

private:
    ReactorState state;
    double dt;

    const double lambda_i = 2.87e-5; // s^-1
    const double lambda_x = 2.09e-5; // s^-1
    const double gamma_i = 0.0639;   // Fission yield for Iodine
    const double gamma_x = 0.0023;   // Fission yield for Xenon
    const double sigma_x = 2.6e-18;  // Xenon absorption cross-section (cm^2)
    const double sigma_f = 0.1;      // Macroscopic fission cross-section (cm^-1)

    std::vector<double> derivatives(double flux, double I, double X); // RK4
};

#endif