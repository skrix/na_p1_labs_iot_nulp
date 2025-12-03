import copy

# Лабораторна робота №3. Системи нелінійних рівнянь.
# Варіант 12.
# Система рівнянь:
# 1) x1^2 + x2^2 + 0.1 - x1 = 0
# 2) 2*x1*x2 - 0.1 - x2 = 0
# Початкове наближення: x1 = 0, x2 = 0
# Точність: e = 10^-5 %
# Методи:
# 1. e-алгоритм (q=2, p=2)
# 2. Метод Ньютона (Гаус: вибір головного елемента по всій матриці)
# 3. Метод січних (Гаус: вибір головного елемента по рядку)

EPSILON_PERCENT = 1e-5  # Задана відносна похибка у %
X0 = [0.0, 0.0]         # Початкове наближення
Q_VAL = 2               # Параметр q для е-алгоритму
P_VAL = 2               # Параметр p для е-алгоритму (початкові ітерації)


def equations_F(x):
    """
    Вектор функцій F(X).
    f1 = x1^2 + x2^2 + 0.1 - x1
    f2 = 2*x1*x2 - 0.1 - x2
    """
    f1 = x[0]**2 + x[1]**2 + 0.1 - x[0]
    f2 = 2*x[0]*x[1] - 0.1 - x[1]
    return [f1, f2]

def jacobian_analytic(x):
    """
    Аналітична матриця Якобі для методу Ньютона.
    df1/dx1 = 2*x1 - 1;  df1/dx2 = 2*x2
    df2/dx1 = 2*x2;      df2/dx2 = 2*x1 - 1
    """
    n = len(x)
    J = [[0.0]*n for _ in range(n)]

    J[0][0] = 2*x[0] - 1
    J[0][1] = 2*x[1]

    J[1][0] = 2*x[1]
    J[1][1] = 2*x[0] - 1

    return J

def iteration_function_G(x):
    """
    Перетворення F(X)=0 -> X=G(X) для методу простої ітерації (та е-алгоритму).
    Потрібно виразити x1, x2 так, щоб забезпечити збіжність (|G'| < 1).

    З рівняння 1: x1 = x1^2 + x2^2 + 0.1
    З рівняння 2: 2*x1*x2 - x2 = 0.1 => x2*(2*x1 - 1) = 0.1 => x2 = 0.1 / (2*x1 - 1)
    """
    new_x = [0.0] * len(x)

    # x1 = g1(x)
    new_x[0] = x[0]**2 + x[1]**2 + 0.1

    # x2 = g2(x)
    denom = 2*x[0] - 1
    if abs(denom) < 1e-14: denom = 1e-14 # Захист від ділення на нуль
    new_x[1] = 0.1 / denom

    return new_x


def solve_gauss(matrix_A, vector_B, pivot_type):
    """
    Розв'язування СЛАР Ax = B методом Гауса.
    pivot_type згідно варіанту:
      'full' - вибір головного елемента по всій матриці (Метод Ньютона).
      'row'  - вибір головного елемента по рядку (Метод Січних).
    """
    n = len(vector_B)
    A = copy.deepcopy(matrix_A)
    B = copy.deepcopy(vector_B)

    # Масив для збереження порядку змінних (потрібен при перестановці стовпців)
    col_order = list(range(n))

    # Прямий хід
    for k in range(n):
        pivot_row = k
        pivot_col = k
        max_val = 0.0

        if pivot_type == 'full':
            # Пошук max по всій підматриці A[k:n, k:n]
            for i in range(k, n):
                for j in range(k, n):
                    if abs(A[i][j]) > abs(max_val):
                        max_val = A[i][j]
                        pivot_row = i
                        pivot_col = j

            # Перестановка рядків
            A[k], A[pivot_row] = A[pivot_row], A[k]
            B[k], B[pivot_row] = B[pivot_row], B[k]

            # Перестановка стовпців
            for row in range(n):
                A[row][k], A[row][pivot_col] = A[row][pivot_col], A[row][k]
            # Запам'ятовуємо зміну порядку змінних
            col_order[k], col_order[pivot_col] = col_order[pivot_col], col_order[k]

        elif pivot_type == 'row':
            # Пошук max тільки в поточному рядку k серед стовпців j >= k
            for j in range(k, n):
                if abs(A[k][j]) > abs(max_val):
                    max_val = A[k][j]
                    pivot_col = j

            # Перестановка стовпців (щоб max елемент рядка став на діагональ)
            for row in range(n):
                A[row][k], A[row][pivot_col] = A[row][pivot_col], A[row][k]
            col_order[k], col_order[pivot_col] = col_order[pivot_col], col_order[k]

            # Перевірка на 0 на діагоналі після перестановки
            if abs(A[k][k]) < 1e-14:
                 return None

        else:
            # Стандартний (по стовпцю) - не використовується у цьому варіанті
            pass

        # Нормування
        pivot = A[k][k]
        if abs(pivot) < 1e-14: return None # Матриця вироджена

        for j in range(k, n):
            A[k][j] /= pivot
        B[k] /= pivot

        # Виключення
        for i in range(k + 1, n):
            factor = A[i][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            B[i] -= factor * B[k]

    # Зворотний хід
    X_temp = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * X_temp[j] for j in range(i + 1, n))
        X_temp[i] = B[i] - sum_ax

    # Відновлення порядку змінних (якщо були перестановки стовпців)
    X_final = [0.0] * n
    for i in range(n):
        # X_temp[i] відповідає змінній з індексом col_order[i]
        X_final[col_order[i]] = X_temp[i]

    return X_final

def check_convergence_relative(x_new, x_old, epsilon_percent):
    """
    Перевірка умови збіжності: |(xi - xi_old)/xi| * 100% < epsilon
    """
    for i in range(len(x_new)):
        numerator = abs(x_new[i] - x_old[i])
        denominator = abs(x_new[i]) if abs(x_new[i]) > 1e-14 else 1.0

        # Якщо значення дуже близьке до 0, використовуємо абсолютну похибку
        if abs(x_new[i]) < 1e-14:
            if numerator * 100.0 >= epsilon_percent: return False
        else:
            if (numerator / denominator) * 100.0 >= epsilon_percent:
                return False
    return True

# --- 3. РЕАЛІЗАЦІЯ МЕТОДІВ ---

def method_epsilon_algorithm(x0, eps_pct, p, q):
    """
    Векторний epsilon-алгоритм.
    Використовує ітераційну формулу X = G(X) для генерації послідовності Sn,
    потім застосовує правило ромба (Wynn's epsilon algorithm) з інверсією Самельсона.
    """
    print(f"\n=== 1. E-алгоритм (q={q}, p={p}, eps={eps_pct}%) ===")
    m = len(x0)
    n_seq = 2 * q + 1  # Кількість членів послідовності Sn (якщо q=m=2, то n=5)

    # 3D масив e[j][k][i], де:
    # j - індекс зсуву послідовності
    # k - індекс epsilon-стовпця (0 відповідає k=-1 в теорії, 1 -> k=0 (Sn), 2 -> k=1...)
    # i - компонент вектора (0..m-1)
    # Розмір: [n_seq+1] рядків, [n_seq+1] стовпців, [m] компонент
    e = [[[0.0]*m for _ in range(n_seq + 1)] for _ in range(n_seq + 50)]

    current_x = list(x0)

    # 1. Виконуємо p початкових ітерацій (без запису в таблицю)
    for _ in range(p):
        current_x = iteration_function_G(current_x)

    iteration_count = 0
    MAX_ITERS = 50

    while iteration_count < MAX_ITERS:
        iteration_count += 1

        # 2. Ініціалізація стовпця k=-1 нулями (індекс k_arr=0)
        for j in range(n_seq + 1):
            for i in range(m):
                e[j][0][i] = 0.0

        # 3. Заповнення стовпця k=0 (Sn) (індекс k_arr=1)
        # Перший елемент S0 - це поточне наближення після p ітерацій
        s_temp = list(current_x)
        for i in range(m):
            e[0][1][i] = s_temp[i]

        # Генеруємо S1...Sn
        converged_early = False
        for j in range(1, n_seq + 1):
            s_prev = list(s_temp)
            s_temp = iteration_function_G(s_temp) # Наступний член послідовності

            for i in range(m):
                e[j][1][i] = s_temp[i]

            # Перевірка збіжності на етапі генерації (якщо Sn вже зійшлося)
            if j == 1 and check_convergence_relative(s_temp, s_prev, eps_pct):
                print(f"Збіжність досягнута на етапі генерації Sn.")
                print(f"Результат: {s_temp}")
                print(f"Дельта: {equations_F(s_temp)}")
                return s_temp

        # 4. Екстраполяція (заповнення таблиці)
        # k_arr йде від 1 (відповідає Sn) до n_seq
        # Формула: e[j][k+1] = e[j+1][k-1] + (e[j+1][k] - e[j][k])^-1

        for k_arr in range(1, n_seq): # k_arr - це поточний стовпець, обчислюємо k_arr+1
            for j in range(n_seq - k_arr):
                # Різниця сусідніх елементів у стовпці: V = e[j+1][k] - e[j][k]
                diff_V = [e[j+1][k_arr][i] - e[j][k_arr][i] for i in range(m)]

                # Інверсія Самельсона: V^(-1) = V / sum(Vi^2)
                norm_sq = sum(v*v for v in diff_V)
                if norm_sq < 1e-25: norm_sq = 1e-25

                inv_V = [v / norm_sq for v in diff_V]

                # Обчислення наступного стовпця
                for i in range(m):
                    e[j][k_arr+1][i] = e[j+1][k_arr-1][i] + inv_V[i]

        # Результат екстраполяції знаходиться в вершині ромба e[0][n_seq][...] (індекс k_arr = n_seq)
        extrapolated_x = [e[0][n_seq][i] for i in range(m)]

        print(f"Цикл {iteration_count}: Екстрапольований X = {[round(val, 6) for val in extrapolated_x]}")

        # Умова завершення: порівняння екстрапольованого результату з початковим для цього циклу
        if check_convergence_relative(extrapolated_x, current_x, eps_pct):
            print(f"Результат знайдено (e-алгоритм): {extrapolated_x}")
            print(f"Дельта F(X): {equations_F(extrapolated_x)}")
            return extrapolated_x

        current_x = extrapolated_x # Нове наближення для наступного циклу

    print("Перевищено ліміт ітерацій e-алгоритму.")
    return current_x

def method_newton_standard(x0, eps_pct):
    """
    Стандартний метод Ньютона.
    X(k) = X(k-1) - J^(-1) * F(X(k-1))
    Обернення J виконується методом Гауса з вибором головного елемента ПО ВСІЙ МАТРИЦІ.
    """
    print(f"\n=== 2. Метод Ньютона (Full Pivot, eps={eps_pct}%) ===")
    x = list(x0)
    k_iter = 0
    MAX_ITERS = 100

    while k_iter < MAX_ITERS:
        k_iter += 1

        F_val = equations_F(x)
        J_mat = jacobian_analytic(x)

        # СЛАР: J * dX = -F
        minus_F = [-val for val in F_val]

        # Виклик Гауса з 'full' pivoting
        delta_x = solve_gauss(J_mat, minus_F, pivot_type='full')

        if delta_x is None:
            print("Матриця Якобі вироджена. Метод зупинено.")
            return x

        x_new = [x[i] + delta_x[i] for i in range(len(x))]

        print(f"Ітерація {k_iter}: X = {[round(val, 8) for val in x_new]}")

        if check_convergence_relative(x_new, x, eps_pct):
            print(f"Збіжність досягнута. Результат: {x_new}")
            print(f"Дельта F(X): {equations_F(x_new)}")
            return x_new

        x = x_new

    print("Перевищено ліміт ітерацій Ньютона.")
    return x

def method_secant(x0, eps_pct):
    """
    Метод січних (двокроковий).
    Використовує кінцево-різницеву матрицю Якобі.
    Обернення J виконується методом Гауса з вибором головного елемента ПО РЯДКУ.
    """
    print(f"\n=== 3. Метод Січних (Row Pivot, eps={eps_pct}%) ===")
    m = len(x0)

    # Для методу січних потрібно два наближення: X^(k) та X^(k-1).
    # Оскільки дано тільки одне (0,0), друге генеруємо штучно з малим зміщенням.
    x_older = list(x0)                 # X^(0)
    x_old = [val + 0.01 for val in x0] # X^(1) - штучне

    k_iter = 0
    MAX_ITERS = 100

    while k_iter < MAX_ITERS:
        k_iter += 1

        # Вектор кроку h = x_old - x_older
        h = [x_old[i] - x_older[i] for i in range(m)]

        # Обчислення кінцево-різницевої матриці Якобі (по стовпцях)
        # J_ij = (fi(x_old + h_vector_only_j) - fi(x_old)) / h_j
        J_diff = [[0.0]*m for _ in range(m)]
        F_old = equations_F(x_old)

        for j in range(m):
            # Формуємо тимчасовий вектор x_tilde, де змінено лише j-ту компоненту
            x_tilde = list(x_old)
            step = h[j] if abs(h[j]) > 1e-14 else 1e-5 # Захист від 0 кроку
            x_tilde[j] += step

            F_tilde = equations_F(x_tilde)

            for i in range(m):
                J_diff[i][j] = (F_tilde[i] - F_old[i]) / step

        # Розв'язуємо J * delta_x = -F(x_old)
        minus_F = [-val for val in F_old]

        # Виклик Гауса з 'row' pivoting
        delta_x = solve_gauss(J_diff, minus_F, pivot_type='row')

        if delta_x is None:
             print("Матриця Якобі (різницева) вироджена.")
             return x_old

        x_new = [x_old[i] + delta_x[i] for i in range(m)]

        print(f"Ітерація {k_iter}: X = {[round(val, 8) for val in x_new]}")

        if check_convergence_relative(x_new, x_old, eps_pct):
             print(f"Збіжність досягнута. Результат: {x_new}")
             print(f"Дельта F(X): {equations_F(x_new)}")
             return x_new

        # Зсув історії наближень
        x_older = x_old
        x_old = x_new

    print("Перевищено ліміт ітерацій методу січних.")
    return x_old

if __name__ == "__main__":
    method_epsilon_algorithm(X0, EPSILON_PERCENT, P_VAL, Q_VAL)
    method_newton_standard(X0, EPSILON_PERCENT)
    method_secant(X0, EPSILON_PERCENT)
