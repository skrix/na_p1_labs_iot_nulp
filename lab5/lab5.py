import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

U_MAX = 100.0       # В
F = 50.0            # Гц
R1 = 5.0            # Ом
R2 = 4.0            # Ом
L1 = 0.01           # Гн
C1 = 300e-6         # Ф
C2 = 150e-6         # Ф
T_MAX = 0.2         # с
H = 0.00001         # крок інтегрування (10 мкс)

# Початкові умови: [u_c1, u_c2, i_l]
X0 = [0.0, 0.0, 0.0]

def get_u1(t):
    return U_MAX * math.sin(2 * math.pi * F * t)

def get_derivatives(t, x):
    """
    x[0] -> u_c1
    x[1] -> u_c2
    x[2] -> i_l
    """
    u_c1, u_c2, i_l = x
    u1_val = get_u1(t)

    # Струм через R1
    i1 = (u1_val - u_c1 - u_c2) / R1

    # Система диференціальних рівнянь
    d_uc1 = i1 / C1
    d_uc2 = (i1 - i_l) / C2
    d_il  = (u_c2 - i_l * R2) / L1

    return np.array([d_uc1, d_uc2, d_il])

def solve_explicit_euler():
    t = 0.0
    x = np.array(X0)
    time_vals = []
    u2_vals = []

    while t <= T_MAX:
        u2 = x[2] * R2
        time_vals.append(t)
        u2_vals.append(u2)

        x = x + H * get_derivatives(t, x)
        t += H
    return time_vals, u2_vals

def solve_modified_euler():
    t = 0.0
    x = np.array(X0)
    time_vals = []
    u2_vals = []

    while t <= T_MAX:
        u2 = x[2] * R2
        time_vals.append(t)
        u2_vals.append(u2)

        k1 = get_derivatives(t, x)
        x_predict = x + H * k1
        k2 = get_derivatives(t + H, x_predict)

        x = x + (H / 2.0) * (k1 + k2)
        t += H
    return time_vals, u2_vals

def solve_implicit_euler():
    t = 0.0
    x = np.array(X0)
    time_vals = []
    u2_vals = []

    while t <= T_MAX:
        u2 = x[2] * R2
        time_vals.append(t)
        u2_vals.append(u2)

        t_next = t + H

        # Рівняння: x_new - x_old - h * f(x_new) = 0
        func = lambda x_new: x_new - x - H * get_derivatives(t_next, x_new)

        x = fsolve(func, x)
        t = t_next
    return time_vals, u2_vals


if __name__ == "__main__":
    print("Розрахунок перехідного процесу для Варіанту 12 (Схема 12)...")

    # Розрахунок трьома методами
    t_expl, u_expl = solve_explicit_euler()
    t_mod, u_mod = solve_modified_euler()
    t_impl, u_impl = solve_implicit_euler()

    # Побудова графіків
    plt.figure(figsize=(10, 6))
    plt.plot(t_expl, u_expl, label='Явний Ейлера', linestyle='--', linewidth=1)
    plt.plot(t_mod, u_mod, label='Модифікований Ейлера', linestyle='-', alpha=0.7)
    # Неявний графік може зливатися з модифікованим через високу точність
    plt.plot(t_impl, u_impl, label='Неявний Ейлера', linestyle=':', linewidth=2)

    plt.title(f"Перехідний процес U2 (Схема 12, h={H}с)")
    plt.xlabel("Час t, с")
    plt.ylabel("Напруга U2, В")
    plt.legend()
    plt.grid(True)
    plt.savefig('./result.png')
    plt.show()

    print("Графік побудовано. Кінцеве значення U2 (Mod): {:.4f} В".format(u_mod[-1]))
