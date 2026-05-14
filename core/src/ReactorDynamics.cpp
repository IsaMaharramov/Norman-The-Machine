#include "ReactorDynamics.hpp"
#include <cmath>
#include <vector>

ReactorEngine::ReactorEngine(double dt) : dt(dt)
{
    reset();
}

void ReactorEngine::reset()
{
    // start -> steady-state equilibrium
    state.neutron_flux = 1e13; // Typical thermal flux
    state.iodine_conc = (gamma_i * sigma_f * state.neutron_flux) / lambda_i;
    state.xenon_conc = ((gamma_i + gamma_x) * sigma_f * state.neutron_flux) /
                       (lambda_x + sigma_x * state.neutron_flux);
    state.power_level = 1.0;
}

std::vector<double> ReactorEngine::derivatives(double flux, double I, double X)
{
    // dI/dt = gamma_i * sigma_f * flux - lambda_i * I
    double dI = gamma_i * sigma_f * flux - lambda_i * I;

    // dX/dt = lambda_i * I + gamma_x * sigma_f * flux - (lambda_x + sigma_x * flux) * X
    double dX = (lambda_i * I) + (gamma_x * sigma_f * flux) - (lambda_x + sigma_x * flux) * X;

    return {dI, dX};
}

void ReactorEngine::step(double control_rod_pos)
{
    // In this project, control_rod_pos (0 to 1) directly maps to neutron flux
    // Realistically, rods change reactivity (rho), but we'll start with flux control
    double target_flux = control_rod_pos * 2e13;

    /// RK4 Integration for Iodine and Xenon
    auto k1 = derivatives(state.neutron_flux, state.iodine_conc, state.xenon_conc);
    auto k2 = derivatives(state.neutron_flux, state.iodine_conc + 0.5 * dt * k1[0], state.xenon_conc + 0.5 * dt * k1[1]);
    auto k3 = derivatives(state.neutron_flux, state.iodine_conc + 0.5 * dt * k2[0], state.xenon_conc + 0.5 * dt * k2[1]);
    auto k4 = derivatives(state.neutron_flux, state.iodine_conc + dt * k3[0], state.xenon_conc + dt * k3[1]);

    state.iodine_conc += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    state.xenon_conc += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);

    state.neutron_flux = target_flux;
    state.power_level = state.neutron_flux / 1e13; // normalized power
}