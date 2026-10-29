"""Numerical ODE solvers, extracted from solve_DE.py so they're importable
(and testable) without triggering that script's plotting/printing side effects.

Each method solves the same second-order ODE (expressed as the coupled
first-order system x' = z, z' = 2z + 3x + 1) that solve_DE.py demonstrates.
"""

import numpy as np


def euler_method(t, x, z, h, n):
    """Solves the ODE using the Euler method.

    Args:
        t (np.ndarray): An array of time points.
        x (np.ndarray): An array to store the solution values.
        z (np.ndarray): An array to store the derivative values.
        h (float): The step size.
        n (int): The number of steps.

    Returns:
        np.ndarray: The solution values.
    """
    for i in range(n):
        t[i+1] = t[i] + h
        z[i+1] = z[i] + h * (2 * z[i] + 3 * x[i] + 1)
        x[i+1] = x[i] + h * z[i]
    return x.copy()


def modified_euler_method(t, x, z, h, n):
    """Solves the ODE using the Modified Euler (explicit midpoint RK2) method.

    For the coupled system x' = z, z' = f(x, z), the midpoint estimate for
    each variable must be offset using the *other* variable's own
    derivative: x's midpoint uses z (since x' = z), and z's midpoint uses
    f(x, z) (its own derivative). An earlier version of this function had
    those two swapped -- both offsets used f(x, z) -- which silently broke
    the method's defining second-order accuracy (empirically, the observed
    convergence order was ~0.8, not ~2; see tests/test_convergence.py).

    Args:
        t (np.ndarray): An array of time points.
        x (np.ndarray): An array to store the solution values.
        z (np.ndarray): An array to store the derivative values.
        h (float): The step size.
        n (int): The number of steps.

    Returns:
        np.ndarray: The solution values.
    """
    for i in range(n):
        t[i+1] = t[i] + h
        k1 = 2 * z[i] + 3 * x[i] + 1  # z' at the current point
        z_mid = z[i] + h/2 * k1       # midpoint estimate of z, offset by z's own derivative
        x_mid = x[i] + h/2 * z[i]     # midpoint estimate of x, offset by x's own derivative (= z)
        z[i+1] = z[i] + h * (2 * z_mid + 3 * x_mid + 1)
        x[i+1] = x[i] + h * z_mid
    return x.copy()


def runge_kutta_method(t, x, z, h, n):
    """Solves the ODE using the fourth-order Runge-Kutta method.

    Args:
        t (np.ndarray): An array of time points.
        x (np.ndarray): An array to store the solution values.
        z (np.ndarray): An array to store the derivative values.
        h (float): The step size.
        n (int): The number of steps.

    Returns:
        tuple: A tuple containing the solution values and the derivative values.
    """
    for i in range(n):
        h2 = h / 2
        t[i+1] = t[i] + h
        m1 = z[i]
        k1 = 2 * z[i] + 3 * x[i] + 1
        m2 = z[i] + h2 * k1
        k2 = 2 * (z[i] + h2 * k1) + 3 * (x[i] + h2 * m1) + 1
        m3 = z[i] + h2 * k2
        k3 = 2 * (z[i] + h2 * k2) + 3 * (x[i] + h2 * m2) + 1
        m4 = z[i] + h * k3
        k4 = 2 * (z[i] + h * k3) + 3 * (x[i] + h * m3) + 1
        z[i+1] = z[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i+1] = x[i] + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    return x.copy(), z.copy()


def adams_moulton_method(t, x, z, h, n):
    """Solves the ODE using one AB4 predictor / AM4 corrector (PECE) step.

    The first 4 points (indices 0-3) must already be filled in by the caller
    (typically with Runge-Kutta) before calling this, since the method needs
    4 prior points to start.

    Args:
        t (np.ndarray): An array of time points.
        x (np.ndarray): An array to store the solution values.
        z (np.ndarray): An array to store the derivative values.
        h (float): The step size.
        n (int): The number of steps.

    Returns:
        np.ndarray: The solution values.
    """
    x_p = x.copy()
    z_p = z.copy()
    for i in range(3, n):
        t[i+1] = t[i] + h
        z_p[i+1] = z[i] + h/24 * (55 * (2 * z[i] + 3 * x[i] + 1) - 59 * (2 * z[i-1] + 3 * x[i-1] + 1) + 37 * (2 * z[i-2] + 3 * x[i-2] + 1) - 9 * (2 * z[i-3] + 3 * x[i-3] + 1))
        x_p[i+1] = x[i] + h/24 * (55 * z[i] - 59 * z[i-1] + 37 * z[i-2] - 9 * z[i-3])
        z[i+1] = z[i] + h/24 * (9 * (2 * z_p[i+1] + 3 * x_p[i+1] + 1) + 19 * (2 * z[i] + 3 * x[i] + 1) - 5 * (2 * z[i-1] + 3 * x[i-1] + 1) + (2 * z[i-2] + 3 * x[i-2] + 1))
        x[i+1] = x[i] + h/24 * (9 * z_p[i+1] + 19 * z[i] - 5 * z[i-1] + z[i-2])
    return x.copy()


def exact_solution(t_list):
    """The known closed-form solution to the demonstration ODE, for comparison."""
    return np.array([
        1/12 * np.exp(-3 - t) * (15 * np.exp(4) + np.exp(4*t) - 4 * np.exp(3 + t))
        for t in t_list
    ])
