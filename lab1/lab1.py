# Лабораторна робота №1.
# ПРЯМІ ТА ІТЕРАЦІЙНІ МЕТОДИ РОЗВ’ЯЗУВАННЯ
# СИСТЕМ ЛІНІЙНИХ АЛГЕБРАЇЧНИХ РІВНЯНЬ
# Варіант 12.
# Завдання:
# Знайти обернену матрицю методом Гауса з
# вибором головного елемента по всій матриці
import copy

k = 12      # Номер завдання 12
s = 0.02*k  # Коефіцієнт для варіанту матриці

# Вхідні дані: Матриця №1
matrix = [
    [8.30, 2.62+s, 4.10, 1.90],
    [3.92, 8.45, 7.78-s, 2.46],
    [3.77, 7.21+s, 8.04, 2.28],
    [2.21, 3.65-s, 1.69, 6.69]
]

def gauss_full_pivot_solve(A_in, B_in):
    """
    Реалізація методу Гауса з вибором головного елемента по всій матриці.
    Алгоритм згідно з Методичкою №1, стор. 13-14 .
    """
    n = len(A_in)

    # 1. Ініціалізація вектора перестановок inx
    inx = list(range(n))

    # 2. Прямий хід. Копіювання матриць A->V, B->P
    V = copy.deepcopy(A_in)
    P = list(B_in)

    # Матриця C для коефіцієнтів (верхня трикутна)
    C = [[0.0] * n for _ in range(n)]
    # Вектор Y для зберігання результатів прямого ходу
    Y = [0.0] * n

    # Цикл по k
    for k in range(n):
        # --- Матричне сортування (пошук головного елемента) ---
        max_val = 0.0
        h = k
        w = k

        for l in range(k, n):
            for f in range(k, n):
                if abs(V[l][f]) > max_val:
                    max_val = abs(V[l][f])
                    h = l
                    w = f

        # Перестановка рядків (h <-> k) у V та P
        # value = P_k; P_k = P_h; P_h = value
        P[k], P[h] = P[h], P[k]

        # value = V_kd; V_kd = V_hd; V_hd = value (для d = 1..n)
        for d in range(n):
            V[k][d], V[h][d] = V[h][d], V[k][d]

        # Перестановка стовпців (w <-> k) у inx, C та V
        # z = inx_k; inx_k = inx_w; inx_w = z
        inx[k], inx[w] = inx[w], inx[k]

        for d in range(n):
            if d < k:
                # якщо d < k -> міняємо у C
                C[d][k], C[d][w] = C[d][w], C[d][k]
            else:
                # інакше -> міняємо у V
                V[d][k], V[d][w] = V[d][w], V[d][k]

        # --- Обчислення (Елімінація) ---
        # Y_k = P_k / V_kk
        Y[k] = P[k] / V[k][k]

        # для i = k+1..n: P_i = P_i - V_ik * Y_k
        for i in range(k + 1, n):
            P[i] = P[i] - V[i][k] * Y[k]

        # для j = k+1..n
        for j in range(k + 1, n):
            # C_kj = V_kj / V_kk
            C[k][j] = V[k][j] / V[k][k]
            # V_ij = V_ij - V_ik * C_kj
            for i in range(k + 1, n):
                V[i][j] = V[i][j] - V[i][k] * C[k][j]

    # --- Обернений хід ---
    X = [0.0] * n
    # X_n = Y_n (у псевдокоді це в циклі, але база рекурсії - останній елемент)

    # Цикл від n-1 до 0
    for i in range(n - 1, -1, -1):
        sum_cx = 0.0
        for j in range(i + 1, n):
            sum_cx += C[i][j] * X[j]
        X[i] = Y[i] - sum_cx

    # --- Впорядкування X_i ---
    # Відновлення порядку коренів згідно з вектором inx
    for i in range(n):
        if inx[i] != i:
            z = inx[i]
            val = X[i]
            X[i] = X[z]
            X[z] = val

            # Оновлення вектора inx (swap logic з методички)
            inx[i] = inx[z]
            inx[z] = z

    return X

def inverse_matrix_gauss_full(A):
    """
    Знаходження оберненої матриці методом Гауса.
    Алгоритм згідно з Методичкою №1, п. 5
    з використанням схеми повного вибору головного елемента
    """
    n = len(A)
    # Матриця для результату INVERS
    INV = [[0.0] * n for _ in range(n)]

    # Для кожного стовпця b одиничної матриці E
    for b in range(n):
        # Формуємо стовпець одиничної матриці E_b
        E_vec = [0.0] * n
        E_vec[b] = 1.0

        # Розв'язуємо систему AX = E_b методом Гауса
        # (в даному випадку - з повним вибором головного елемента)
        X_col = gauss_full_pivot_solve(A, E_vec)

        # Записуємо результат у відповідний стовпець оберненої матриці
        for i in range(n):
            INV[i][b] = X_col[i]

    return INV

# Виконання програми
print(f"Вхідна матриця (k={k}, s={s:.2f}):")
for row in matrix:
    print([round(x, 4) for x in row])

inverse_m = inverse_matrix_gauss_full(matrix)

print("\nОбернена матриця:")
for row in inverse_m:
    print(" ".join(f"{x: .6f}" for x in row))

# Перевірка: A * A_inv повинна дорівнювати одиничній матриці E
print("\nПеревірка (A * A_inv):")
check = [[0.0]*4 for _ in range(4)]
for i in range(4):
    for j in range(4):
        val = sum(matrix[i][k] * inverse_m[k][j] for k in range(4))
        check[i][j] = val
for row in check:
    print(" ".join(f"{x: .6f}" for x in row))
