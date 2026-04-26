import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, lu_factorization
from plots import plot_planes_3d

st.set_page_config(page_title="Factorización LU", layout="wide")
st.title("📐 Método Directo: Factorización LU")

st.markdown("""
**Método Directo.** Descompone la matriz A en el producto:
$$A = L \\cdot U$$
donde **L** es triangular inferior (con 1s en la diagonal) y **U** es triangular superior.
Luego resuelve el sistema en dos fases:
1. **Sustitución hacia adelante:** $Ly = b$
2. **Sustitución hacia atrás:** $Ux = y$

> En el contexto de la quiroptocoria, este método proporciona la **solución exacta de referencia**
> con la que se comparan todos los métodos iterativos.
""")

st.divider()

# Selector de escenario
scenario_name = st.selectbox("Escenario:", list(SCENARIOS.keys()), key="lu_esc")
scenario = SCENARIOS[scenario_name]
A = scenario["A"].copy()
b = scenario["b"].copy()

st.markdown(f"**{scenario['desc']}**")
st.markdown(f"κ(A) = `{np.linalg.cond(A):.4e}`")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Matriz A:**")
    st.dataframe(pd.DataFrame(A, columns=["x₁","x₂","x₃"],
                               index=["Ec.1","Ec.2","Ec.3"]).round(4),
                 use_container_width=True)
with col2:
    st.markdown("**Vector b:**")
    st.dataframe(pd.DataFrame(b, index=["b₁","b₂","b₃"], columns=["valor"]).round(4),
                 use_container_width=True)

if st.button("▶ Ejecutar Factorización LU", type="primary"):
    L, U, x, steps = lu_factorization(A, b)

    if x is None:
        st.error(f"Error: {steps[0]['desc']}")
        st.stop()

    st.subheader("🔢 Paso a paso — Eliminación hacia adelante")
    st.markdown("En cada paso se muestra cómo evolucionan L y U:")

    for idx, step in enumerate(steps):
        with st.expander(f"Paso {idx+1}: {step['desc']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**L** (triangular inferior)")
                st.dataframe(
                    pd.DataFrame(step["L"].round(6),
                                 columns=["col 1","col 2","col 3"],
                                 index=["fila 1","fila 2","fila 3"]),
                    use_container_width=True,
                )
            with c2:
                st.markdown("**U** (triangular superior)")
                st.dataframe(
                    pd.DataFrame(step["U"].round(6),
                                 columns=["col 1","col 2","col 3"],
                                 index=["fila 1","fila 2","fila 3"]),
                    use_container_width=True,
                )

    st.subheader("➡️ Sustitución hacia adelante: Ly = b")
    y = np.zeros(3)
    for i in range(3):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
        st.markdown(
            f"y[{i+1}] = (b[{i+1}] − Σ L[{i+1},j]·y[j]) / L[{i+1},{i+1}] = **{y[i]:.8f}**"
        )

    st.subheader("⬅️ Sustitución hacia atrás: Ux = y")
    x_calc = np.zeros(3)
    for i in range(2, -1, -1):
        x_calc[i] = (y[i] - U[i, i+1:] @ x_calc[i+1:]) / U[i, i]
        st.markdown(
            f"x[{i+1}] = (y[{i+1}] − Σ U[{i+1},j]·x[j]) / U[{i+1},{i+1}] = **{x_calc[i]:.8f}**"
        )

    st.divider()
    st.subheader("✅ Resultado Final")

    col_r1, col_r2, col_r3 = st.columns(3)
    names = ["x₁ Tránsito digestivo", "x₂ Esfuerzo de vuelo", "x₃ Cobertura vegetal"]
    for ci, (col, nm) in enumerate(zip([col_r1, col_r2, col_r3], names)):
        with col:
            st.metric(nm, f"{x_calc[ci]:.8f}")

    res = np.linalg.norm(A @ x_calc - b)
    st.info(f"**Residual ‖Ax − b‖₂ = {res:.2e}**")

    # Verificación con numpy
    x_np = solve_exact(A, b)
    if x_np is not None:
        diff = np.linalg.norm(x_calc - x_np)
        st.success(f"Diferencia con solución de numpy: ‖x_LU − x_np‖ = {diff:.2e}")

    st.subheader("🗺️ Visualización 3D")
    fig = plot_planes_3d(A, b, title=f"LU — {scenario_name}")
    if fig:
        st.pyplot(fig, use_container_width=True)
