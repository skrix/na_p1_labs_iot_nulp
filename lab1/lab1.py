# Завдання:
# Знайти обернену матрицю методом Гауса з
# вибором головного елемента по всій матриці

k = 12     # Номер завдання 12
s = 0.02*k # Коефіцієнт для варіанту матриці

matrix = [
  [8.30, 2.62+s, 4.10, 1.90],
  [3.92, 8.45, 7.78-s, 2.46],
  [3.77, 7.21+s, 8.04, 2.28],
  [2.21, 3.65-s, 1.69, 6.69]
]

def gauss_full_pivot_solve(A, b):
    n = len(A)

    # inx[i] = i  (перестановки невідомих x_i)
    inx = list(range(n))

    # копії A -> V, b -> P; C – матриця коефіцієнтів трикутної системи
    V = [row[:] for row in A]
    P = b[:]
    C = [[0.0] * n for _ in range(n)]

    # --------- Прямий хід з матричним сортуванням ---------
    for k in range(n):
        # матричне сортування: пошук max |V[l,f]| у підматриці k..n-1
        max_val = abs(V[k][k])
        h = k      # рядок головного елемента
        w = k      # стовпець головного елемента
        for l in range(k, n):
            for f in range(k, n):
                if abs(V[l][f]) > max_val:
                    max_val = abs(V[l][f])
                    h = l
                    w = f

        # перестановка рядків h <-> k у V і P
        if h != k:
            V[k], V[h] = V[h], V[k]
            P[k], P[h] = P[h], P[k]

        # перестановка стовпців w <-> k, оновлення inx і C / V
        if w != k:
            # вектор inx зберігає, який x_j опинився на позиції k
            inx[k], inx[w] = inx[w], inx[k]
            for d in range(n):
                if d < k:
                    # вже обчислені елементи C міняємо місцями
                    C[d][k], C[d][w] = C[d][w], C[d][k]
                else:
                    # для ще не опрацьованих рядків переставляємо стовпці у V
                    V[d][k], V[d][w] = V[d][w], V[d][k]

        # крок звичайного Гауса (як у загальному алгоритмі)
        Yk = P[k] / V[k][k]
        P[k] = Yk                  # P_k := Y_k
        for i in range(k + 1, n):  # оновлення правих частин
            P[i] -= V[i][k] * Yk
        for j in range(k + 1, n):  # формування C та обнулення піддіагоналі
            C[k][j] = V[k][j] / V[k][k]
            for i in range(k + 1, n):
                V[i][j] -= V[i][k] * C[k][j]

    Y = P[:]  # після прямого ходу у P зберігається вектор Y

    # --------- Зворотний хід ---------
    X = [0.0] * n
    X[n - 1] = Y[n - 1]
    for i in range(n - 2, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += C[i][j] * X[j]
        X[i] = Y[i] - s

    # --------- Впорядкування x_i за допомогою inx ---------
    X_ord = [0.0] * n
    for pos in range(n):
        orig = inx[pos]      # який початковий x_orig стоїть на позиції pos
        X_ord[orig] = X[pos] # значення X_pos належить x_orig

    return X_ord

def inverse_gauss_full_pivot(A):
    n = len(A)
    INV = [[0.0] * n for _ in range(n)]

    # одинична матриця E
    for b in range(n):
        e = [0.0] * n
        e[b] = 1.0          # стовпець E_ib
        x = gauss_full_pivot_solve(A, e)
        for i in range(n):
            INV[i][b] = x[i]
    return INV


matrix_inversed = inverse_gauss_full_pivot(matrix)
for row in matrix_inversed:
    print(" ".join(f"{x: .6f}" for x in row))
