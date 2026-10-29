import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numerical_methods import (
    euler_method,
    modified_euler_method,
    runge_kutta_method,
    adams_moulton_method,
    exact_solution,
)


def show_plot(t, ans_rightDecision, ans_numericalMethod, labels, name):
    """Displays a plot comparing the exact solution with a numerical method.

    Args:
        t (list): A list of time points.
        ans_rightDecision (list): A list of the exact solution values.
        ans_numericalMethod (list): A list of the numerical solution values.
        labels (list): A list of labels for the plot legends.
        name (str): The title of the plot.
    """
    plt.figure()
    plt.plot(t, ans_rightDecision, 'o-k', alpha = 0.7, label = labels[0], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.plot(t, ans_numericalMethod, 'h-m', alpha = 0.7, label = labels[1], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.xlabel('Value of t')
    plt.ylabel('Value of x')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()

def print_info(t, answer, name, z='nothing'):
    """Prints the coordinates and solution values.

    Args:
        t (list): A list of time points.
        answer (list): A list of the solution values.
        name (str): The name of the method.
        z (str, optional): Additional values to print. Defaults to 'nothing'.
    """
    print(f'\\n\\n---------- {name} ----------')
    print(f'Coordinates t: {t}')
    # print(f"Values y'(t): {z}")
    print(f'Value x(t): {answer}')
    print('#'*60)

def calc_absolute_error(x, y, n):
    """Calculates the absolute error between two lists of values.

    Args:
        x (list): A list of the first set of values.
        y (list): A list of the second set of values.
        n (int): The number of points to compare.

    Returns:
        list: A list of the absolute errors.
    """
    result = [abs(x[i]-y[i]) for i in range(n+1)]
    return result

def create_only_plot_error(t, error, labels, name):
    """Displays a plot of the absolute error.

    Args:
        t (list): A list of time points.
        error (list): A list of the absolute error values.
        labels (str): The label for the plot legend.
        name (str): The title of the plot.
    """
    plt.figure()
    plt.plot(t, error, 'o-k', alpha = 0.7, label = labels, lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.xlabel('Value of t')
    plt.ylabel('Error')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()
    print('\\n\\n')

def show_all_plots(t, ans_rightDecision, ans_numericalMethod, labels, name):
    """Displays a plot comparing the exact solution with multiple numerical methods.

    Args:
        t (list): A list of time points.
        ans_rightDecision (list): A list of the exact solution values.
        ans_numericalMethod (list): A list of lists of the numerical solution values.
        labels (list): A list of labels for the plot legends.
        name (str): The title of the plot.
    """
    plt.figure(figsize=(9,5))
    plt.plot(t, ans_rightDecision, 'o-k', alpha = 0.7, label = labels[0], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.plot(t, ans_numericalMethod[0], 'h-m', alpha = 0.7, label = labels[1], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.plot(t, ans_numericalMethod[1], 'x--b', alpha = 0.7, label = labels[2], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.plot(t, ans_numericalMethod[2], '+-.r', alpha = 0.7, label = labels[3], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.plot(t, ans_numericalMethod[3], 'd:g', alpha = 0.7, label = labels[4], lw = 2, mec = 'k', mew = 3, ms = 3)
    plt.xlabel('Value of t')
    plt.ylabel('Value of x')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()

t0 = 1
x0 = 1
z0 = -1
tn = 3
h = 0.2

# Use whole steps only; the final grid point never exceeds tn.
n = int(np.floor((tn - t0) / h + 1e-12))
t_list = t0 + np.arange(n + 1) * h
z = np.zeros([n+1])
x = np.zeros([n+1])
t = np.zeros([n+1])
z[0] = z0
x[0] = x0
t[0] = t0

'''Exact solution of the ODE'''
ans_rightDecision = exact_solution(t_list)  # retain precision for error measurements

print_info(t_list, ans_rightDecision, name = "Exact solution of the ODE")

ans_eulerMethod = euler_method(t.copy(), x.copy(), z.copy(), h, n)
print_info(t_list, ans_eulerMethod, name="Euler method solution")
show_plot(t_list, ans_rightDecision, ans_eulerMethod, labels=["Built-in method", "Euler method"], name="Euler method solution")

ans_modifiedEulerMethod = modified_euler_method(t.copy(), x.copy(), z.copy(), h, n)
print_info(t_list, ans_modifiedEulerMethod, name="Modified Euler method solution")
show_plot(t_list, ans_rightDecision, ans_modifiedEulerMethod, labels=["Built-in method", "modified Euler method"], name="Modified Euler")

ans_rungeKuttaMethod, z_rungeKuttaMethod = runge_kutta_method(t.copy(), x.copy(), z.copy(), h, n)
print_info(t_list, ans_rungeKuttaMethod, name="Runge-Kutta method solution")
show_plot(t_list, ans_rightDecision, ans_rungeKuttaMethod, labels=["Built-in method", "Runge–Kutta method"], name="Runge-Kutta method solution")

z = np.zeros([n+1])
z_p = np.zeros([n+1])
x = np.zeros([n+1])
x_p = np.zeros([n+1])
t = np.zeros([n+1])
z[:4] = z_rungeKuttaMethod[:4]
z_p[:4] = z_rungeKuttaMethod[:4]
x[:4] = ans_rungeKuttaMethod[:4]
x_p[:4] = ans_rungeKuttaMethod[:4]
t[:4] = t_list[:4]

ans_adamsMoultonMethod = adams_moulton_method(t.copy(), x.copy(), z.copy(), h, n)
print_info(t_list, ans_adamsMoultonMethod, name="Adams-Moulton method solution")
show_plot(t_list, ans_rightDecision, ans_adamsMoultonMethod, labels=["Built-in method", "Adams-Moulton method"], name="Adams-Moulton method solution")

error_eulerMethod = calc_absolute_error(ans_rightDecision, ans_eulerMethod, n)
error_modifiedEulerMethod = calc_absolute_error(ans_rightDecision, ans_modifiedEulerMethod, n)
error_rungeKuttaMethod = calc_absolute_error(ans_rightDecision, ans_rungeKuttaMethod, n)
error_adamsMoultonMethod = calc_absolute_error(ans_rightDecision, ans_adamsMoultonMethod, n);

create_only_plot_error(t_list, error = error_eulerMethod, labels =  "Euler method", name = "Absolute error of the Euler method")
create_only_plot_error(t_list, error = error_modifiedEulerMethod, labels = "modified Euler method", name = "Absolute error of the modified Euler method")
create_only_plot_error(t_list, error = error_rungeKuttaMethod, labels = "Runge–Kutta method", name = "Absolute error of the Runge-Kutta method")
create_only_plot_error(t_list, error = error_adamsMoultonMethod, labels = "Adams-Moulton method", name = "Absolute error of the Adams-Moulton method")

data = { 't': t_list,
        'Exact solution': ans_rightDecision,
        'Euler method': ans_eulerMethod,
        'Modified Euler method': ans_modifiedEulerMethod,
        'Runge-Kutta method': ans_rungeKuttaMethod,
        'Adams-Moulton method': ans_adamsMoultonMethod
        }
np.round(pd.DataFrame(data), 4)

data = { 't': t_list,
        'Δ Euler method': error_eulerMethod,
        'Δ modified Euler method': error_modifiedEulerMethod,
        'Δ Runge-Kutta method': error_rungeKuttaMethod,
        'Δ Adams-Moulton method': error_adamsMoultonMethod
        }
np.round(pd.DataFrame(data), 4)

solutions = [ans_eulerMethod, ans_modifiedEulerMethod, ans_rungeKuttaMethod, ans_adamsMoultonMethod]
labels = ['Exact solution', 'Euler method', 'Modified Euler method', 'Runge-Kutta method', 'Adams-Moulton method']
show_all_plots(t_list, ans_rightDecision, solutions, labels, name='Comparison of solutions')
