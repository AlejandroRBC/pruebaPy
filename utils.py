import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Sistema por defecto ──────────────────────────────────────────────────────
DEFAULT_A = [[4, -1, 0],
             [-1,  4, -1],
             [0, -1,  4]]

DEFAULT_B = [15, 10, 10]

def parse_matrix(rows_text: list[str], n: int):
    """Convierte lista de strings a np.array."""
    A = []
    for row in rows_text:
        vals = list(map(float, row.split()))
        if len(vals) != n:
            raise ValueError(f"Se esperaban {n} valores por fila.")
        A.append(vals)
    return np.array(A, dtype=float)

def plot_planes(A, b):
    """Grafica los 3 planos y la solución del sistema 3x3."""
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    x_range = np.linspace(sol[0] - 4, sol[0] + 4, 30)
    y_range = np.linspace(sol[1] - 4, sol[1] + 4, 30)
    X, Y = np.meshgrid(x_range, y_range)

    colors = ['#4e9af1', '#f1914e', '#6ef14e']
    labels = ['Ecuación 1', 'Ecuación 2', 'Ecuación 3']

    for i in range(3):
        a0, a1, a2 = A[i]
        if abs(a2) > 1e-10:
            Z = (b[i] - a0 * X - a1 * Y) / a2
            ax.plot_surface(X, Y, Z, alpha=0.35, color=colors[i], label=labels[i])
        elif abs(a1) > 1e-10:
            Z_range = np.linspace(sol[2] - 4, sol[2] + 4, 30)
            Xp, Zp = np.meshgrid(x_range, Z_range)
            Yp = (b[i] - a0 * Xp) / a1
            ax.plot_surface(Xp, Yp, Zp, alpha=0.35, color=colors[i], label=labels[i])

    ax.scatter(*sol, color='red', s=80, zorder=5, label=f'Solución ({sol[0]:.2f}, {sol[1]:.2f}, {sol[2]:.2f})')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('x₃')
    ax.set_title('Intersección de planos')
    ax.legend(loc='upper left', fontsize=7)
    plt.tight_layout()
    return fig

def input_system(key_prefix="default", default_A=None, default_b=None):
    """Widget reutilizable para ingresar el sistema Ax=b (3x3)."""
    import streamlit as st

    if default_A is None:
        default_A = DEFAULT_A
    if default_b is None:
        default_b = DEFAULT_B

    st.subheader("Sistema de ecuaciones Ax = b")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Matriz A** (3 filas, valores separados por espacio)")
        rows = []
        for i in range(3):
            default_row = " ".join(map(str, default_A[i]))
            row = st.text_input(f"Fila {i+1}", value=default_row, key=f"{key_prefix}_row{i}")
            rows.append(row)

    with col2:
        st.markdown("**Vector b**")
        b_vals = []
        for i in range(3):
            val = st.number_input(f"b[{i+1}]", value=float(default_b[i]), key=f"{key_prefix}_b{i}")
            b_vals.append(val)

    try:
        A = parse_matrix(rows, 3)
        b = np.array(b_vals)
        return A, b, True
    except Exception as e:
        st.error(f"Error al leer el sistema: {e}")
        return None, None, False
