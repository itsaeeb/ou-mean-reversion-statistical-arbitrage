"""
Unit Tests for Phase 4: OU Process Simulation and Numerical Validation
======================================================================
"""

import unittest
import numpy as np
import sys
import os

test_dir = os.path.dirname(os.path.abspath(__file__))
phase4_root = os.path.abspath(os.path.join(test_dir, ".."))
scripts_dir = os.path.join(phase4_root, "scripts")
sys.path.insert(0, scripts_dir)

from ou_simulation import (
    exact_ou_simulation,
    euler_maruyama_ou_simulation,
    simulate_coupled_paths,
    check_euler_stability,
    theoretical_moments,
    estimate_ou_parameters_from_path
)


class TestPhase4OUSimulation(unittest.TestCase):
    
    def setUp(self):
        self.theta = 10.0
        self.mu = 0.50
        self.sigma = 0.20
        self.x0 = 1.00
        self.dt = 1.0 / 252.0
        self.T = 1.0

    def test_exact_ou_shape_and_initial_condition(self):
        """Verify output dimensions, time grid, and initial condition X_0."""
        num_paths = 5
        t_grid, paths = exact_ou_simulation(
            self.theta, self.mu, self.sigma, self.x0, self.T, self.dt,
            num_paths=num_paths, random_state=42
        )
        expected_N = int(round(self.T / self.dt))
        self.assertEqual(len(t_grid), expected_N + 1)
        self.assertEqual(paths.shape, (num_paths, expected_N + 1))
        np.testing.assert_allclose(paths[:, 0], self.x0)
        self.assertAlmostEqual(t_grid[0], 0.0)
        self.assertAlmostEqual(t_grid[-1], self.T, places=5)

    def test_exact_ou_deterministic_decay(self):
        """Verify zero-noise simulation exactly tracks deterministic ODE solution."""
        N = int(round(self.T / self.dt))
        Z_zero = np.zeros((1, N))
        t_grid, paths = exact_ou_simulation(
            self.theta, self.mu, self.sigma, self.x0, self.T, self.dt,
            num_paths=1, innovations=Z_zero
        )
        expected = self.mu + (self.x0 - self.mu) * np.exp(-self.theta * t_grid)
        np.testing.assert_allclose(paths[0], expected, rtol=1e-12)

    def test_euler_deterministic_step(self):
        """Verify Euler step with zero noise matches (1 - theta*dt)*X_t + theta*mu*dt."""
        N = int(round(self.T / self.dt))
        Z_zero = np.zeros((1, N))
        t_grid, paths, is_stable = euler_maruyama_ou_simulation(
            self.theta, self.mu, self.sigma, self.x0, self.T, self.dt,
            num_paths=1, innovations=Z_zero
        )
        self.assertTrue(is_stable)
        # Check first step manually
        expected_x1 = self.x0 + self.theta * (self.mu - self.x0) * self.dt
        self.assertAlmostEqual(paths[0, 1], expected_x1, places=12)

    def test_coupled_simulation_and_error_scaling(self):
        """Verify coupled exact and Euler simulations share driving noise and error grows with dt."""
        # Fine step size
        _, ex_fine, eu_fine, _ = simulate_coupled_paths(
            self.theta, self.mu, self.sigma, self.x0, 0.5, 1.0/252.0, num_paths=10, random_state=42
        )
        mae_fine = np.mean(np.abs(ex_fine - eu_fine))
        
        # Coarse step size
        _, ex_coarse, eu_coarse, _ = simulate_coupled_paths(
            self.theta, self.mu, self.sigma, self.x0, 0.5, 21.0/252.0, num_paths=10, random_state=42
        )
        mae_coarse = np.mean(np.abs(ex_coarse - eu_coarse))
        
        # Coarser dt should produce strictly greater discretisation error
        self.assertGreater(mae_coarse, mae_fine)

    def test_euler_stability_detection(self):
        """Verify Euler stability function correctly identifies stable and unstable regimes."""
        # Stable: theta = 5.0, dt = 0.1 -> theta*dt = 0.5 < 2
        stab1 = check_euler_stability(5.0, 0.1)
        self.assertTrue(stab1["is_stable"])
        self.assertEqual(stab1["stability_regime"], "Monotonically Stable")
        
        # Oscillatory Stable: theta = 15.0, dt = 0.1 -> theta*dt = 1.5 in (1, 2)
        stab2 = check_euler_stability(15.0, 0.1)
        self.assertTrue(stab2["is_stable"])
        self.assertTrue("Oscillatory" in stab2["stability_regime"])
        
        # Explosive: theta = 25.0, dt = 0.1 -> theta*dt = 2.5 > 2
        stab3 = check_euler_stability(25.0, 0.1)
        self.assertFalse(stab3["is_stable"])
        self.assertTrue("Unstable" in stab3["stability_regime"])

    def test_theoretical_moments_identities(self):
        """Verify theoretical expectation, variance, and stationary sigma_eq."""
        t_eval = np.array([0.0, 0.5, 1.0, 5.0])
        mom = theoretical_moments(self.theta, self.mu, self.sigma, self.x0, t_eval)
        
        # At t = 0: mean = x0, var = 0
        self.assertAlmostEqual(mom["mean"][0], self.x0, places=12)
        self.assertAlmostEqual(mom["variance"][0], 0.0, places=12)
        
        # Asymptotics
        expected_asymp_var = self.sigma ** 2 / (2.0 * self.theta)
        self.assertAlmostEqual(mom["stationary_variance"], expected_asymp_var, places=12)
        self.assertAlmostEqual(mom["sigma_eq"], np.sqrt(expected_asymp_var), places=12)
        
        # Half-life: ln(2) / theta
        self.assertAlmostEqual(mom["half_life_years"], np.log(2.0) / self.theta, places=12)
        self.assertAlmostEqual(mom["half_life_days"], mom["half_life_years"] * 252.0, places=12)

    def test_parameter_recovery_on_synthetic_path(self):
        """Verify AR(1) OLS recovery of known parameters on a long synthetic path."""
        theta_true = 8.0
        mu_true = 0.25
        sigma_true = 0.15
        
        # Simulate long path (N = 10,000)
        t_g, p = exact_ou_simulation(
            theta=theta_true, mu=mu_true, sigma=sigma_true, x0=mu_true,
            T=40.0, dt=self.dt, num_paths=1, random_state=42
        )
        res = estimate_ou_parameters_from_path(p[0], dt=self.dt)
        self.assertTrue(res["is_valid"])
        
        # Parameter estimates should be within tight confidence interval of ground truth
        self.assertAlmostEqual(res["theta"], theta_true, delta=0.8)
        self.assertAlmostEqual(res["mu"], mu_true, delta=0.03)
        self.assertAlmostEqual(res["sigma"], sigma_true, delta=0.02)


if __name__ == "__main__":
    unittest.main()
