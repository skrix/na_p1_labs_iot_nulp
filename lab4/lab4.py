import math

# Лабораторна робота №4. Числове інтегрування.
# Варіант 12.
# Функція: f(x) = sqrt(x^2 + 3) / x^2
# Проміжок: [1, 2]

def f(x):
    """Підінтегральна функція"""
    return math.sqrt(x**2 + 3) / (x**2)

def F(x):
    """Первісна функції (для точного розрахунку)"""
    return - (math.sqrt(x**2 + 3) / x) + math.log(x + math.sqrt(x**2 + 3))

def exact_integral(a, b):
    """Формула Ньютона-Лейбніца (1.2)"""
    return F(b) - F(a)

# ---------------------------------------------------------
# 1. МЕТОД СЕРЕДНІХ ПРЯМОКУТНИКІВ
# ---------------------------------------------------------
def method_middle_rectangles(a, b, n):
    integral = 0
    h = (b - a) / n

    # Цикл сумування значень у середніх точках
    for i in range(n):
        # x_mid - середина i-го відрізка: x_i + h/2
        x_mid = a + i * h + h / 2
        integral = integral + f(x_mid)

    # Остаточне множення на крок h
    result = integral * h
    return result

# ---------------------------------------------------------
# 2. МЕТОД ТРАПЕЦІЙ
# ---------------------------------------------------------
def method_trapezoidal(a, b, n):
    integral = 0
    h = (b - a) / n
    x = a + h
    fa = f(a)
    fb = f(b)

    for i in range(1, n):
        integral = integral + f(x)
        x = x + h

    result = h * ((fa + fb) / 2 + integral)
    return result

# ---------------------------------------------------------
# 3. МЕТОД СІМПСОНА
# ---------------------------------------------------------
def method_simpson(a, b, m):
    integral = 0
    n = 2 * m  # Кількість розбиттів має бути парною
    h = (b - a) / n
    fa = f(a)
    fb = f(b)

    # Перший цикл (для непарних індексів: 1, 3, ..., 2m-1)
    # Множник 4
    for i in range(1, m + 1):
        x = a + (2 * i - 1) * h
        integral = integral + 4 * f(x)

    # Другий цикл (для парних індексів: 2, 4, ..., 2m-2)
    # Множник 2
    for i in range(1, m):
        x = a + 2 * i * h
        integral = integral + 2 * f(x)

    result = (h / 3) * (fa + fb + integral)
    return result

# ---------------------------------------------------------
# Головна частина програми
# ---------------------------------------------------------
def main():
    # Межі інтегрування
    a = 1.0
    b = 2.0

    # Точне значення
    exact_val = exact_integral(a, b)
    print(f"Точне значення (формула Ньютона-Лейбніца): {exact_val:.10f}")
    print("-" * 60)
    print(f"{'Метод':<30} | {'Результат':<12} | {'Похибка':<12}")
    print("-" * 60)

    # 1. Метод середніх прямокутників (n=30)
    n_rect = 30
    res_rect = method_middle_rectangles(a, b, n_rect)
    err_rect = abs(res_rect - exact_val)
    print(f"{'Середніх прямокутників (n=30)':<30} | {res_rect:.10f} | {err_rect:.2e}")

    # 2. Метод трапецій (n=30)
    n_trap = 30
    res_trap = method_trapezoidal(a, b, n_trap)
    err_trap = abs(res_trap - exact_val)
    print(f"{'Трапецій (n=30)':<30} | {res_trap:.10f} | {err_trap:.2e}")

    # 3. Метод Сімпсона (m=10 -> n=20)
    m_simpson = 10
    res_simp = method_simpson(a, b, m_simpson)
    err_simp = abs(res_simp - exact_val)
    print(f"{'Сімпсона (m=10, n=20)':<30} | {res_simp:.10f} | {err_simp:.2e}")
    print("-" * 60)

if __name__ == "__main__":
    main()
