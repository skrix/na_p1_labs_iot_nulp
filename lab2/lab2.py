# Лабораторна робота №2. Методи уточнення коренів нелінійних рівнянь.
# Варіант 12.
# Рівняння: f(x) = ln(x) + x = 0
# Похідна: f'(x) = 1 + 1/x
# Проміжок: [0.1, 16]

import math

def f(x: float) -> float:
    """
    Функція f(x) = ln(x) + x
    """
    # Захист від недопустимих значень (ln(x) не існує для x <= 0)
    if x <= 0:
        raise ValueError(f"Недопустиме значення аргументу для логарифма: x = {x}")
    return math.log(x) + x

def df(x: float) -> float:
    """
    Похідна f'(x) = 1 + 1/x
    """
    if x == 0:
        raise ValueError("Ділення на нуль у похідній")
    return 1.0 + 1.0 / x

def bisection_with_localization(a_start: float, h: float, eps: float):
    print(f"\n--- 1. Метод поділу ділянки навпіл з пошуком локалізації ---")
    print(f"Початкова точка a: {a_start}, Крок h: {h}, Epsilon: {eps}")

    # Крок 1: Пошук ділянки локалізації
    a = a_start
    b = a + h

    try:
        fa = f(a)
        fb = f(b)
    except ValueError as e:
        print(f"Помилка при обчисленні функції: {e}")
        return

    # Перевірка напрямку
    # Якщо модуль функції зростає і знаки однакові, змінюємо напрямок
    if abs(fb) > abs(fa) and (fa * fb > 0):
        print("Зміна напрямку пошуку (h = -h)")
        h = -1 * h
        b = a + h
        fb = f(b)

    # Цикл пошуку інтервалу, де функція змінює знак
    iterations_loc = 0
    while fa * fb > 0:
        a = b
        b = a + h
        fa = f(a)
        fb = f(b)
        iterations_loc += 1
        if iterations_loc > 1000:
            print("Не вдалося локалізувати корінь за розумну кількість кроків.")
            return

    print(f"Корінь локалізовано на відрізку: [{min(a,b):.4f}, {max(a,b):.4f}]")

    # Крок 2: Ітераційне уточнення
    left = min(a, b)
    right = max(a, b)
    f_left = f(left)

    x = 0
    k = 0
    while True:
        k += 1
        x = (left + right) / 2.0
        fx = f(x)

        # Умова збіжності (|fx| < eps)
        if abs(fx) < eps:
            break

        if fx * f_left > 0:
            left = x
            f_left = fx
        else:
            right = x

    print(f"Результат:")
    print(f"x = {x:.8f}")
    print(f"f(x) = {f(x):.8e}")
    print(f"Кількість ітерацій уточнення: {k}")
    return x

def bisection_classic(a: float, b: float, eps: float):
    print(f"\n--- 2. Метод поділу ділянки навпіл (класичний) ---")
    print(f"Відрізок: [{a}, {b}], Epsilon: {eps}")

    if f(a) * f(b) > 0:
        print("Помилка: На заданому проміжку функція не змінює знак.")
        return

    # Умова збіжності: b - a < 2 * eps
    k = 0
    # Змінні для меж
    curr_a = a
    curr_b = b
    fa = f(curr_a)

    while (curr_b - curr_a) >= 2 * eps:
        k += 1
        x = (curr_a + curr_b) / 2.0
        fx = f(x)

        if fx * fa > 0:
            curr_a = x
            fa = fx  # Оновлюємо значення функції на лівій межі
        else:
            curr_b = x

    # Результат - середина останнього відрізка
    x_res = (curr_a + curr_b) / 2.0

    print(f"Результат:")
    print(f"x = {x_res:.8f}")
    print(f"f(x) = {f(x_res):.8e}")
    print(f"Кількість ітерацій: {k}")
    return x_res

def simple_iteration(x0: float, a_interval: float, b_interval: float, eps_percent: float):
    print(f"\n--- 3. Метод простої ітерації ---")
    print(f"Початкове наближення x0: {x0}, Epsilon (%): {eps_percent}%")

    # 1. Приведення до вигляду x = x + alpha * f(x)
    # Вибір alpha згідно стор. 24 (формули 6.6 - 6.7)
    # alpha = -1 / max|f'(x)|, якщо f'(x) > 0

    # Аналіз похідної: f'(x) = 1 + 1/x.
    # На проміжку [0.1, 16] похідна монотонно спадає.
    # Максимум похідної досягається в точці a_interval (0.1).
    max_df = df(a_interval)

    # Оскільки f'(x) > 0 на всьому проміжку, alpha має бути від'ємним
    alpha = -1.0 / max_df

    print(f"Розраховане значення alpha: {alpha:.6f} (для max|f'(x)| = {max_df:.4f})")

    # Перевірка умови збіжності |1 + alpha*f'(x)| < 1 (Теоретична перевірка)
    # g'(x) = 1 + alpha * f'(x).
    # Оскільки alpha = -1/max_df, то min(g'(x)) = 0, max(g'(x)) < 1. Збіжність гарантована.

    x_old = x0
    k = 0

    while True:
        k += 1

        # Формула ітерації: x = x + alpha * f(x)
        # Це еквівалент x = g(x)
        x_new = x_old + alpha * f(x_old)

        # Умова збіжності (стор. 25): |(x - x_old)/x| * 100% < eps
        if x_new == 0: # Захист від ділення на нуль
            rel_error = abs(x_new - x_old) * 100
        else:
            rel_error = abs((x_new - x_old) / x_new) * 100

        if rel_error < eps_percent:
            break

        x_old = x_new

        if k > 10000:
            print("Перевищено ліміт ітерацій!")
            break

    print(f"Результат:")
    print(f"x = {x_new:.8f}")
    print(f"f(x) = {f(x_new):.8e}")
    print(f"Відносна похибка: {rel_error:.6e} %")
    print(f"Кількість ітерацій: {k}")
    return x_new

if __name__ == "__main__":
    # Загальні параметри
    interval_start = 0.1
    interval_end = 16.0

    # Для методів поділу
    EPS_ABS = 1e-4

    # Для методу простої ітерації
    EPS_REL_PERCENT = 0.01 # 0.01%

    # 1. Метод поділу ділянки навпіл з пошуком локалізації
    # Починаємо пошук з лівого краю інтервалу, крок h беремо довільний, наприклад 0.5
    bisection_with_localization(a_start=interval_start, h=0.5, eps=EPS_ABS)

    # 2. Метод поділу ділянки навпіл (класичний)
    # Використовуємо весь заданий проміжок [0.1, 16]
    # Примітка: для коректної роботи методу на кінцях проміжку функція має мати різні знаки.
    # f(0.1) = ln(0.1) + 0.1 ≈ -2.3 + 0.1 < 0
    # f(16) = ln(16) + 16 > 0
    # Умова виконується.
    bisection_classic(a=interval_start, b=interval_end, eps=EPS_ABS)

    # 3. Метод простої ітерації
    # Початкове наближення беремо з інтервалу, наприклад x0 = 0.5
    simple_iteration(x0=0.5, a_interval=interval_start, b_interval=interval_end, eps_percent=EPS_REL_PERCENT)
