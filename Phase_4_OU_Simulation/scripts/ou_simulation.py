"""
Phase 4: OU Process Simulation and Numerical Engine
===================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This module provides reusable numerical methods for:
1. Exact discretisation of the Ornstein-Uhlenbeck (OU) continuous-time SDE:
      dX_t = theta * (mu - X_t) * dt + sigma * dW_t
   Exact transition:
      X_(t+Delta) = mu * (1 - exp(-theta*Delta)) + exp(-theta*Delta) * X_t
                   + sigma * sqrt((1 - exp(-2*theta*Delta)) / (2*theta)) * Z_t
      where Z_t ~ i.i.d. N(0, 1).
2. Euler-Maruyama discrete approximation:
      X_(t+Delta) = X_t + theta * (mu - X_t) * Delta + sigma * sqrt(Delta) * Z_t
3. Coupled pathwise simulation sharing the exact same Brownian increments Z_t.
4. Linear numerical stability analysis of Euler-Maruyama (|1 - theta * Delta| < 1).
5. Theoretical time-dependent and stationary moments of the OU process.
6. Analytical parameter recovery via exact AR(1) OLS (identical to Phase 3).
"""

import math
import numpy as np
import pandas as pd


def exact_ou_simulation(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    dt: float = 1.0 / 252.0,
    num_paths: int = 1,
    random_state: int = None,
    innovations: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate OU paths using exact Gaussian discretisation.
    
    Parameters:
    -----------
    theta : float
        Mean-reversion speed (annualized, theta > 0).
    mu : float
        Long-term equilibrium mean.
    sigma : float
        Diffusion volatility (annualized, sigma > 0).
    x0 : float
        Initial spread value at t = 0.
    T : float
        Total time horizon in years.
    dt : float
        Discrete time step (default 1/252 yr).
    num_paths : int
        Number of independent paths to simulate.
    random_state : int, optional
        Seed for reproducibility.
    innovations : np.ndarray, optional
        Pre-generated standard normal innovations of shape (num_paths, N).
        If provided, guarantees identical Brownian driving noise for coupling.
        
    Returns:
    --------
    time_grid : np.ndarray of shape (N + 1,)
    paths : np.ndarray of shape (num_paths, N + 1)
    """
    if theta <= 0:
        raise ValueError(f"theta must be strictly positive, got {theta}")
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    if T <= 0:
        raise ValueError(f"T must be strictly positive, got {T}")
        
    N = int(round(T / dt))
    time_grid = np.linspace(0.0, N * dt, N + 1)
    
    # Exact AR(1) parameters
    b = np.exp(-theta * dt)
    a = mu * (1.0 - b)
    var_eta = (sigma ** 2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * dt))
    sigma_eta = np.sqrt(max(var_eta, 0.0))
    
    if innovations is None:
        rng = np.random.RandomState(random_state)
        Z = rng.normal(0.0, 1.0, size=(num_paths, N))
    else:
        if innovations.shape != (num_paths, N):
            raise ValueError(f"innovations shape {innovations.shape} does not match ({num_paths}, {N})")
        Z = innovations
        
    paths = np.empty((num_paths, N + 1), dtype=float)
    paths[:, 0] = x0
    
    # Vectorized step-by-step transition across all paths
    for k in range(N):
        paths[:, k + 1] = a + b * paths[:, k] + sigma_eta * Z[:, k]
        
    return time_grid, paths


def euler_maruyama_ou_simulation(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    dt: float = 1.0 / 252.0,
    num_paths: int = 1,
    random_state: int = None,
    innovations: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Simulate OU paths using Euler-Maruyama numerical discretization:
       X_(k+1) = X_k + theta * (mu - X_k) * dt + sigma * sqrt(dt) * Z_k
               = (1 - theta * dt) * X_k + theta * mu * dt + sigma * sqrt(dt) * Z_k
               
    Returns:
    --------
    time_grid : np.ndarray of shape (N + 1,)
    paths : np.ndarray of shape (num_paths, N + 1)
    is_stable : bool (False if explosive growth or NaN detected)
    """
    if theta <= 0:
        raise ValueError(f"theta must be strictly positive, got {theta}")
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
        
    N = int(round(T / dt))
    time_grid = np.linspace(0.0, N * dt, N + 1)
    
    amp_factor = 1.0 - theta * dt
    drift_const = theta * mu * dt
    diff_scale = sigma * np.sqrt(dt)
    
    if innovations is None:
        rng = np.random.RandomState(random_state)
        Z = rng.normal(0.0, 1.0, size=(num_paths, N))
    else:
        if innovations.shape != (num_paths, N):
            raise ValueError(f"innovations shape {innovations.shape} does not match ({num_paths}, {N})")
        Z = innovations
        
    paths = np.empty((num_paths, N + 1), dtype=float)
    paths[:, 0] = x0
    
    is_stable = True
    for k in range(N):
        paths[:, k + 1] = amp_factor * paths[:, k] + drift_const + diff_scale * Z[:, k]
        # Check for numerical explosion
        if not np.all(np.isfinite(paths[:, k + 1])) or np.any(np.abs(paths[:, k + 1]) > 1e8):
            is_stable = False
            paths[:, k + 1:] = np.nan
            break
            
    return time_grid, paths, is_stable


def simulate_coupled_paths(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    dt: float = 1.0 / 252.0,
    num_paths: int = 1,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Simulate exact and Euler-Maruyama OU paths simultaneously using the EXACT
    SAME underlying Brownian driving noise Z ~ N(0, 1).
    
    This guarantees that path differences reflect purely numerical discretisation
    error rather than sample variance from mismatched random paths.
    """
    N = int(round(T / dt))
    rng = np.random.RandomState(random_state)
    Z = rng.normal(0.0, 1.0, size=(num_paths, N))
    
    t_grid, exact_paths = exact_ou_simulation(
        theta=theta, mu=mu, sigma=sigma, x0=x0, T=T, dt=dt,
        num_paths=num_paths, innovations=Z
    )
    
    _, euler_paths, is_stable = euler_maruyama_ou_simulation(
        theta=theta, mu=mu, sigma=sigma, x0=x0, T=T, dt=dt,
        num_paths=num_paths, innovations=Z
    )
    
    return t_grid, exact_paths, euler_paths, is_stable


def check_euler_stability(theta: float, dt: float) -> dict:
    """
    Evaluate the linear numerical stability of the Euler-Maruyama scheme
    applied to the Ornstein-Uhlenbeck SDE:
       X_(k+1) = (1 - theta * dt) * X_k + ...
       
    Stability criteria:
    - Amplification factor g = 1 - theta * dt
    - Strictly Stable: |1 - theta * dt| < 1  <=>  0 < theta * dt < 2  <=>  dt < 2 / theta
    - Monotonically Stable: 0 <= 1 - theta * dt < 1  <=>  0 < theta * dt <= 1  <=>  dt <= 1 / theta
    - Oscillatory Stable: -1 < 1 - theta * dt < 0  <=>  1 < theta * dt < 2
    - Explosive / Unstable: theta * dt >= 2  <=>  |g| >= 1
    """
    theta_dt = theta * dt
    amp_factor = 1.0 - theta_dt
    crit_dt = 2.0 / theta
    
    if theta_dt < 1.0:
        regime = "Monotonically Stable"
        is_stable = True
    elif theta_dt < 2.0:
        regime = "Oscillatory Stable (Sub-optimal)"
        is_stable = True
    elif np.isclose(theta_dt, 2.0):
        regime = "Marginally Unstable (Periodic Oscillation)"
        is_stable = False
    else:
        regime = "Linearly Unstable / Explosive (|g| > 1)"
        is_stable = False
        
    return {
        "theta": theta,
        "dt": dt,
        "theta_dt": theta_dt,
        "amplification_factor": amp_factor,
        "critical_dt": crit_dt,
        "is_stable": is_stable,
        "stability_regime": regime
    }


def theoretical_moments(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    t: np.ndarray | float
) -> dict:
    """
    Calculate the exact theoretical mean, variance, standard deviation,
    and asymptotic stationary distribution of the Ornstein-Uhlenbeck process.
    
    Formulas:
    E[X_t | X_0] = mu + (x0 - mu) * exp(-theta * t)
    Var(X_t | X_0) = (sigma^2 / (2 * theta)) * (1 - exp(-2 * theta * t))
    Stationary distribution: X_inf ~ N(mu, sigma_eq^2), sigma_eq = sigma / sqrt(2*theta)
    Half-life: t_1/2 = ln(2) / theta
    """
    t_arr = np.asarray(t, dtype=float)
    exp_neg_theta_t = np.exp(-theta * t_arr)
    
    mean_t = mu + (x0 - mu) * exp_neg_theta_t
    var_t = (sigma ** 2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t_arr))
    std_t = np.sqrt(np.maximum(var_t, 0.0))
    
    stationary_var = sigma ** 2 / (2.0 * theta)
    sigma_eq = np.sqrt(stationary_var)
    half_life_years = np.log(2.0) / theta
    half_life_days = half_life_years * 252.0
    
    return {
        "t": t_arr,
        "mean": mean_t,
        "variance": var_t,
        "std": std_t,
        "stationary_mean": mu,
        "stationary_variance": stationary_var,
        "sigma_eq": sigma_eq,
        "half_life_years": half_life_years,
        "half_life_days": half_life_days
    }


def estimate_ou_parameters_from_path(
    path: np.ndarray,
    dt: float = 1.0 / 252.0
) -> dict:
    """
    Recover continuous OU parameters (theta, mu, sigma, sigma_eq, half-life)
    from a 1D discrete time series path using the Phase 3 exact AR(1) OLS methodology.
    """
    N = len(path)
    if N < 5:
        return {"is_valid": False, "failure_reason": "Insufficient observations (< 5)"}
        
    Y = path[1:]
    X = path[:-1]
    M = len(Y)
    
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    x_dev = X - x_mean
    y_dev = Y - y_mean
    
    sum_xx = np.sum(x_dev ** 2)
    sum_xy = np.sum(x_dev * y_dev)
    
    if sum_xx <= 1e-14 or not np.isfinite(sum_xx):
        return {"is_valid": False, "failure_reason": "Zero or infinite regressor variance"}
        
    b = sum_xy / sum_xx
    a = y_mean - b * x_mean
    
    residuals = Y - (a + b * X)
    sse = np.sum(residuals ** 2)
    dof = max(M - 2, 1)
    res_var = sse / dof
    
    if b >= 1.0:
        return {"is_valid": False, "failure_reason": f"b >= 1 (b = {b:.6f})", "ar1_b": b}
    if b <= 0.0:
        return {"is_valid": False, "failure_reason": f"b <= 0 (b = {b:.6f})", "ar1_b": b}
    if res_var <= 0.0 or not np.isfinite(res_var):
        return {"is_valid": False, "failure_reason": "Non-positive residual variance"}
        
    theta_hat = -np.log(b) / dt
    mu_hat = a / (1.0 - b)
    sigma_eq_hat = np.sqrt(res_var / (1.0 - b ** 2))
    sigma_hat = sigma_eq_hat * np.sqrt(2.0 * theta_hat)
    half_life_days = np.log(2.0) / (-np.log(b))
    half_life_years = half_life_days * dt
    
    return {
        "ar1_a": a,
        "ar1_b": b,
        "residual_var": res_var,
        "theta": theta_hat,
        "mu": mu_hat,
        "sigma": sigma_hat,
        "sigma_eq": sigma_eq_hat,
        "half_life_days": half_life_days,
        "half_life_years": half_life_years,
        "is_valid": True,
        "failure_reason": "VALID"
    }
