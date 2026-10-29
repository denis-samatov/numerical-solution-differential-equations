"""Convergence-order tests for the numerical ODE solvers.

Each method has a known theoretical order of accuracy. We solve the same
problem solve_DE.py demonstrates (with a known closed-form exact solution)
at two step sizes, h and h/2, and check that halving the step size reduces
the global error by roughly the factor the method's order predicts:

    error(h) / error(h/2) ≈ 2^order

This is standard numerical-methods practice for empirically verifying an
implementation's order of accuracy.
"""

import numpy as np
import pytest

from numerical_methods import (
    euler_method,
    modified_euler_method,
    runge_kutta_method,
    adams_moulton_method,
    exact_solution,
)

T0, X0, Z0, TN = 1.0, 1.0, -1.0, 3.0


def _global_error(method, h):
    n = int(round((TN - T0) / h))
    t_list = np.linspace(T0, T0 + n * h, n + 1)
    t = np.zeros(n + 1)
    x = np.zeros(n + 1)
    z = np.zeros(n + 1)
    t[0], x[0], z[0] = T0, X0, Z0

    result = method(t.copy(), x.copy(), z.copy(), h, n)
    x_numeric = result[0] if isinstance(result, tuple) else result

    exact = exact_solution(t_list)
    return np.max(np.abs(exact - x_numeric))


def _adams_moulton_global_error(h):
    n = int(round((TN - T0) / h))
    t_list = np.linspace(T0, T0 + n * h, n + 1)
    t = np.zeros(n + 1)
    x = np.zeros(n + 1)
    z = np.zeros(n + 1)
    t[0], x[0], z[0] = T0, X0, Z0

    # Adams-Moulton needs 4 starting points; bootstrap with Runge-Kutta,
    # matching solve_DE.py's own setup.
    x_rk, z_rk = runge_kutta_method(t.copy(), x.copy(), z.copy(), h, n)
    exact = exact_solution(t_list)

    t2 = np.zeros(n + 1)
    x2 = np.zeros(n + 1)
    z2 = np.zeros(n + 1)
    t2[:4] = t_list[:4]
    x2[:4] = x_rk[:4]
    z2[:4] = z_rk[:4]

    x_numeric = adams_moulton_method(t2, x2, z2, h, n)
    return np.max(np.abs(exact - x_numeric))


@pytest.mark.parametrize(
    "method, expected_order, tolerance",
    [
        (euler_method, 1, 0.3),
        (modified_euler_method, 2, 0.3),
        (runge_kutta_method, 4, 0.5),
    ],
)
def test_observed_convergence_order(method, expected_order, tolerance):
    h = 0.05
    error_h = _global_error(method, h)
    error_h_half = _global_error(method, h / 2)

    observed_order = np.log2(error_h / error_h_half)

    assert observed_order == pytest.approx(expected_order, abs=tolerance), (
        f"{method.__name__}: expected order ~{expected_order}, "
        f"observed {observed_order:.3f} (error(h)={error_h:.2e}, error(h/2)={error_h_half:.2e})"
    )


def test_adams_moulton_converges_and_outperforms_euler():
    # The 4-step Adams-Moulton predictor-corrector's exact theoretical order
    # is more sensitive to the bootstrap step for a demonstration problem
    # this small, so we assert the more robust, weaker property the audit
    # actually needs: it converges as h shrinks, and comfortably
    # outperforms the first-order Euler method at the same step size.
    h = 0.05
    error_h = _adams_moulton_global_error(h)
    error_h_half = _adams_moulton_global_error(h / 2)

    assert error_h_half < error_h, "halving the step size should reduce the global error"

    euler_error = _global_error(euler_method, h)
    assert error_h < euler_error, "Adams-Moulton should be far more accurate than Euler at the same step size"


def test_predictor_corrector_observed_order():
    # Coarser h=0.05 lies outside the asymptotic regime for this problem.
    errors = [_adams_moulton_global_error(h) for h in (0.0125, 0.00625, 0.003125)]
    for coarse, fine in zip(errors, errors[1:]):
        assert np.log2(coarse / fine) == pytest.approx(4.0, abs=0.3)


@pytest.mark.parametrize("method, order", [(euler_method, 1), (modified_euler_method, 2), (runge_kutta_method, 4)])
def test_second_analytic_initial_value_problem_checks_x_and_z(method, order):
    # Same documented system, independent initial values and closed-form oracle.
    # w=x+1/3 has eigenvalues3,-1, so w=c1*exp(3t)+c2*exp(-t).
    x0, z0, end = 2.0, 0.5, 1.0
    c1 = (z0 + x0 + 1/3) / 4
    c2 = (3 * (x0 + 1/3) - z0) / 4
    errors = []
    for h in (0.02, 0.01, 0.005):
        n = round(end / h)
        times = np.arange(n + 1) * h
        t, x, z = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
        x[0], z[0] = x0, z0
        method(t, x, z, h, n)
        exact_x = c1 * np.exp(3 * times) + c2 * np.exp(-times) - 1/3
        exact_z = 3 * c1 * np.exp(3 * times) - c2 * np.exp(-times)
        errors.append((np.max(np.abs(x - exact_x)), np.max(np.abs(z - exact_z))))
    for coarse, fine in zip(errors, errors[1:]):
        for coarse_component, fine_component in zip(coarse, fine):
            assert np.log2(coarse_component / fine_component) == pytest.approx(order, abs=0.3)
