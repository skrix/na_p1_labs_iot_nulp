import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R1 = 11.0
R2 = 22.0
R3 = 33.0
# R4 в умові відсутній, вважаємо розрив кола (нескінченний опір)
C1 = 34e-6  # 34 мкФ
C2 = 26e-6  # 26 мкФ

L_MAX = 8.3
L_MIN = 0.83
I_MIN = 1.0
I_MAX = 2.0

a = 0.02      # параметр часу
T = 2 * a     # Період
T_SIM = 5 * T # Час моделювання (5 періодів)
POINTS_PER_PERIOD = 100
h = T / 400   # Крок інтегрування (згідно вимог <= T/400)


def get_u1(t):
    """
    Формування вхідної напруги u1(t).
    Графік: Трикутна форма (пилкоподібна або трикутна).
    Трактування 'a a' та T=2a:
    0 -> a/2: зростання 0 -> 10
    a/2 -> 1.5a: спад 10 -> -10
    1.5a -> 2a: зростання -10 -> 0
    """
    t_cycle = t % T # Час в межах одного періоду

    # Амплітуда
    U_peak = 10.0

    # Точки перегину для симетричного трикутника з періодом 2a
    t1 = a / 2
    t2 = 3 * a / 2

    if t_cycle <= t1:
        # Лінійне зростання: y = kx => 10 = k(a/2) => k = 20/a
        return (U_peak / t1) * t_cycle
    elif t_cycle <= t2:
        # Лінійний спад від 10 до -10
        # y - 10 = k(x - a/2). k = -20/a
        return U_peak - (2 * U_peak / a) * (t_cycle - t1)
    else:
        # Лінійне зростання від -10 до 0
        return -U_peak + (U_peak / (T - t2)) * (t_cycle - t2)

def get_inductance(i_val):
    """
    Апроксимація нелінійної індуктивності L(i).
    Використовується кубічний сплайн на інтервалі [i_min, i_max].
    """
    abs_i = abs(i_val)

    if abs_i <= I_MIN:
        return L_MAX
    elif abs_i >= I_MAX:
        return L_MIN
    else:
        # Розрахунок коефіцієнтів сплайну
        # x1 = I_MIN, x2 = I_MAX, y1 = L_MAX, y2 = L_MIN
        # m1 = 0, m2 = 0 (похідні на кінцях рівні нулю для гладкості)

        x = abs_i
        x1 = I_MIN
        x2 = I_MAX
        h_spline = x2 - x1

        # Базисні функції Ерміта
        # y(x) = (b1*y1 + b2*y2)/h^3 + (b3*m1 + b4*m2)/h^2
        # Оскільки m1=m2=0, друга частина формули зникає.

        b1 = (2 * (x - x1) + h_spline) * (x2 - x)**2
        b2 = (2 * (x2 - x) + h_spline) * (x - x1)**2

        L_spline = (b1 * L_MAX + b2 * L_MIN) / (h_spline**3)
        return L_spline

def system_equations(t, Y):
    """
    Система диференціальних рівнянь.
    Y[0] = uC1
    Y[1] = i2
    Y[2] = uC2
    """
    uc1 = Y[0]
    i2 = Y[1]
    uc2 = Y[2]

    u1 = get_u1(t)

    # Розрахунок проміжної напруги uA (вузол між R1, R3, R2)
    # Рівняння вузла: (u1 - uc1 - uA)/R1 = uA/R3 + i2
    # (u1 - uc1)/R1 - i2 = uA(1/R1 + 1/R3)

    G_eq = (1/R1 + 1/R3)
    term_source = (u1 - uc1) / R1

    uA = (term_source - i2) / G_eq

    # 1. duC1/dt = i1 / C1
    # i1 = (u1 - uc1 - uA) / R1
    i1 = (u1 - uc1 - uA) / R1
    duc1_dt = i1 / C1

    # 2. di2/dt = uL / L(i)
    # uA - i2*R2 - uL - uc2 = 0 => uL = uA - uc2 - i2*R2
    uL = uA - uc2 - i2 * R2
    L_val = get_inductance(i2)
    di2_dt = uL / L_val

    # 3. duC2/dt = iC2 / C2
    # i2 = iC2 (оскільки R4 відсутній/нескінченний)
    # Якщо б R4 був: iC2 = i2 - uc2/R4
    duc2_dt = i2 / C2

    return np.array([duc1_dt, di2_dt, duc2_dt])

def runge_kutta_3_b(t, Y, h):
    """
    Метод Рунге-Кутта 3-го порядку
    Аналог методу Ралстона.
    """
    # K1 = h * f(xn, yn)
    k1 = h * system_equations(t, Y)

    # K2 = h * f(xn + 1/3 h, yn + 1/3 K1)
    k2 = h * system_equations(t + h/3.0, Y + k1/3.0)

    # K3 = h * f(xn + 2/3 h, yn + 2/3 K2)
    k3 = h * system_equations(t + 2.0*h/3.0, Y + 2.0*k2/3.0)

    # Yn+1 = Yn + (K1 + 3K3)/4
    # Перевірка: метод порядку 3, ваги 1/4 і 3/4.
    Y_next = Y + (k1 + 3*k3) / 4.0

    return Y_next


t_values = []
u1_values = []
uc1_values = []
i2_values = []
uc2_values = []
l_values = [] # Для перевірки зміни індуктивності

# Початкові умови [uC1, i2, uC2]
Y = np.array([0.0, 0.0, 0.0])
t = 0.0

print("Початок розрахунку...")
print(f"Крок: {h}, Кількість точок: {int(T_SIM/h)}")

while t <= T_SIM:
    # Збереження результатів
    t_values.append(t)
    u1_values.append(get_u1(t))
    uc1_values.append(Y[0])
    i2_values.append(Y[1])
    uc2_values.append(Y[2])
    l_values.append(get_inductance(Y[1]))

    # Крок інтегрування
    Y = runge_kutta_3_b(t, Y, h)
    t += h

data = {
    'Time': t_values,
    'U1': u1_values,
    'Uc1': uc1_values,
    'I2': i2_values,
    'U2': uc2_values
}
df = pd.DataFrame(data)
df.to_csv('lab6/result.dat', sep='\t', index=False, float_format='%.6f')

print("Розрахунок завершено. Результати збережено в lab6/result.dat")

# --- ПОБУДОВА ГРАФІКІВ ---
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_values, u1_values, label='Вхідна напруга U1(t)', color='blue')
plt.plot(t_values, uc2_values, label='Вихідна напруга U2(t)', color='red', linestyle='--')
plt.title('Графіки напруг')
plt.xlabel('Час, с')
plt.ylabel('Напруга, В')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_values, i2_values, label='Струм індуктивності i2(t)', color='green')
plt.title('Графік струму')
plt.xlabel('Час, с')
plt.ylabel('Струм, А')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_values, uc1_values, label='Напруга на C1', color='orange')
plt.title('Напруга на ємності C1')
plt.xlabel('Час, с')
plt.ylabel('Напруга, В')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
