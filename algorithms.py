import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# ESCENARIOS (matrices del PDF TemaYjustificacion)
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "🌿 Bosque Saludable (Caso Ideal)": {
        "A": np.array([[10, 1, 2],
                       [1,  8, 3],
                       [2,  3, 12]], dtype=float),
        "b": np.array([15, 20, 25], dtype=float),
        "desc": (
            "Ecosistema en equilibrio. La matriz es **simétrica, definida positiva "
            "y estrictamente diagonal dominante**, lo que garantiza convergencia "
            "rápida en todos los métodos iterativos."
        ),
        "cond_label": "Bien condicionado",
    },
    "🔥 Fragmentación del Hábitat (Caso Estrés)": {
        "A": np.array([[45, 12, 18],
                       [12, 38, 20],
                       [18, 20, 50]], dtype=float),
        "b": np.array([85, 90, 110], dtype=float),
        "desc": (
            "Perturbación ambiental severa. Los coeficientes reflejan un aumento "
            "drástico en las interdependencias energéticas durante la migración "
            "forzada por cambio climático."
        ),
        "cond_label": "Moderadamente condicionado",
    },
    "💀 Deforestación Severa (Caso Mal Condicionado)": {
        "A": np.array([[10,     5,     2],
                       [10.001, 5.001, 2],
                       [2,      3,     12]], dtype=float),
        "b": np.array([17, 17.001, 25], dtype=float),
        "desc": (
            "Homogeneización del hábitat. Las filas 1 y 2 son casi proporcionales, "
            "generando **hiperplanos casi paralelos** y un número de condición κ(A) "
            "extremadamente alto. Los métodos iterativos clásicos se estancan."
        ),
        "cond_label": "Mal condicionado",
    },
}

VAR_NAMES = ["x₁ (Tránsito digestivo)", "x₂ (Esfuerzo de vuelo)", "x₃ (Cobertura vegetal)"]

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMOS
# ─────────────────────────────────────────────────────────────────────────────

def solve_exact(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

def jacobi(A, b, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    errors = []
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            if abs(A[i, i]) < 1e-14:
                return x, errors, False, "Pivote nulo"
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        err = np.linalg.norm(x_new - x, np.inf)
        errors.append(err)
        x = x_new.copy()
        if err < tol:
            return x, errors, True, k + 1
    return x, errors, False, max_iter

def gauss_seidel(A, b, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    errors = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-14:
                return x, errors, False, "Pivote nulo"
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i, i]
        err = np.linalg.norm(x - x_old, np.inf)
        errors.append(err)
        if err < tol:
            return x, errors, True, k + 1
    return x, errors, False, max_iter

def sor(A, b, omega=1.25, tol=1e-6, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    errors = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-14:
                return x, errors, False, "Pivote nulo"
            gs = (b[i] - sum(A[i, j] * x[j] for j in range(n) if j != i)) / A[i, i]
            x[i] = (1 - omega) * x[i] + omega * gs
        err = np.linalg.norm(x - x_old, np.inf)
        errors.append(err)
        if err < tol:
            return x, errors, True, k + 1
    return x, errors, False, max_iter

def pcg(A, b, tol=1e-6, max_iter=500):
    """Gradiente Conjugado Precondicionado — Algoritmo Mz de Suñagua (2020).
    Precondicionador: M = diag(A)^{-1} (trivial de Jacobi).
    """
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    # Precondicionador diagonal: Mz = r  →  z = diag(A)^{-1} r
    diag_inv = 1.0 / np.diag(A)
    z = diag_inv * r
    p = z.copy()
    rz_old = r @ z
    errors = []

    for k in range(max_iter):
        Ap = A @ p
        denom = p @ Ap
        if abs(denom) < 1e-30:
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        z = diag_inv * r
        rz_new = r @ z
        err = np.linalg.norm(r)
        errors.append(err)
        if err < tol:
            return x, errors, True, k + 1
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, errors, False, max_iter

def lu_factorization(A, b):
    """Devuelve L, U, x, pasos detallados."""
    n = len(b)
    L = np.eye(n)
    U = A.copy()
    steps = []

    for k in range(n):
        for i in range(k + 1, n):
            if abs(U[k, k]) < 1e-14:
                return None, None, None, [{"desc": f"Pivote nulo en posición ({k+1},{k+1})"}]
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, :] -= factor * U[k, :]
            steps.append({
                "desc": f"Fila {i+1} ← Fila {i+1} − {factor:.4f} · Fila {k+1}",
                "L": L.copy(),
                "U": U.copy(),
            })

    # Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return L, U, x, steps

def run_all_methods(A, b, omega=1.25, tol=1e-6, max_iter=500):
    results = {}
    _, _, x_jac, errors_jac = jacobi(A, b, tol, max_iter)
    conv_jac = len(errors_jac) < max_iter
    results["Jacobi"] = {"x": x_jac, "errors": errors_jac,
                          "iters": len(errors_jac), "conv": conv_jac}

    _, _, x_gs, errors_gs = gauss_seidel(A, b, tol, max_iter)
    conv_gs = len(errors_gs) < max_iter
    results["Gauss-Seidel"] = {"x": x_gs, "errors": errors_gs,
                                 "iters": len(errors_gs), "conv": conv_gs}

    _, _, x_sor, errors_sor = sor(A, b, omega, tol, max_iter)
    conv_sor = len(errors_sor) < max_iter
    results["SOR"] = {"x": x_sor, "errors": errors_sor,
                       "iters": len(errors_sor), "conv": conv_sor}

    _, _, x_pcg, errors_pcg = pcg(A, b, tol, max_iter)
    conv_pcg = len(errors_pcg) < max_iter
    results["PCG (Suñagua)"] = {"x": x_pcg, "errors": errors_pcg,
                                  "iters": len(errors_pcg), "conv": conv_pcg}

    return results

# helper: unpack con firma fija
def _unpack(result_tuple):
    x, errors, conv, info = result_tuple
    return x, errors, conv, info
