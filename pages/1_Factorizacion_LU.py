import streamlit as st
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import input_system, plot_planes

st.set_page_config(page_title="Factorización LU", layout="wide")
st.title("📐 Factorización LU")
st.markdown("""
**Método Directo.** Descompone la matriz A en el producto de dos matrices triangulares:  
`A = L · U`  
donde L es triangular inferior (Lower) y U es triangular superior (Upper).  
Luego resuelve el sistema en dos pasos: **Ly = b** y **Ux = y**.
""")

A, b, ok = input_system(key_prefix="lu")

if ok and st.button("▶ Ejecutar Factorización LU"):
    n = 3
    L = np.eye(n)
    U = A.copy().astype(float)
    steps = []

    # ── Eliminación hacia adelante ──────────────────────────────────────────
    for k in range(n):
        for i in range(k + 1, n):
            if abs(U[k, k]) < 1e-12:
                st.error("Elemento pivote nulo — el método requiere pivoteo.")
                st.stop()
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, :] -= factor * U[k, :]
            steps.append({
                "desc": f"Eliminar columna {k+1} en fila {i+1}: factor = {factor:.4f}",
                "L": L.copy(),
                "U": U.copy()
            })

    st.subheader("Paso a paso")
    st.markdown("**Matrices L y U tras cada operación de eliminación:**")

    for idx, step in enumerate(steps):
        with st.expander(f"Paso {idx+1}: {step['desc']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**L (triangular inferior)**")
                st.dataframe(
                    np.round(step['L'], 4),
                    hide_index=False,
                    use_container_width=True
                )
            with c2:
                st.markdown("**U (triangular superior)**")
                st.dataframe(
                    np.round(step['U'], 4),
                    hide_index=False,
                    use_container_width=True
                )

    # ── Sustitución ─────────────────────────────────────────────────────────
    st.subheader("Sustitución hacia adelante: Ly = b")
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        st.markdown(f"y[{i+1}] = `{y[i]:.6f}`")

    st.subheader("Sustitución hacia atrás: Ux = y")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        st.markdown(f"x[{i+1}] = `{x[i]:.6f}`")

    st.success(f"✅ Solución: x₁={x[0]:.6f},  x₂={x[1]:.6f},  x₃={x[2]:.6f}")

    # ── Verificación ────────────────────────────────────────────────────────
    residual = np.linalg.norm(A @ x - b)
    st.info(f"Residual ‖Ax - b‖ = {residual:.2e}")

    # ── Gráfico ─────────────────────────────────────────────────────────────
    st.subheader("Visualización 3D — Intersección de planos")
    fig = plot_planes(A, b)
    if fig:
        st.pyplot(fig)
