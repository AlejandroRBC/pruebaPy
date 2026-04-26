import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, gauss_seidel
from plots import plot_planes_3d, METHOD_COLORS

st.set_page_config(page_title="Gauss-Seidel", layout="wide")
st.title("🔁 Método Iterativo: Gauss-Seidel")

st.markdown("""
**Método Iterativo Clásico.** A diferencia de Jacobi, Gauss-Seidel actualiza cada variable
**inmediatamente** usando los valores más recientes calculados en la misma iteración:

$$x_i^{(k+1)} = \\frac{1}{a_{ii}}\\left(b_i - \\sum_{j<i} a_{ij}\\,x_j^{(k+1)} - \\sum_{j>i} a_{ij}\\,x_j^{(k)}\\right)$$

**Ventaja sobre Jacobi:** Generalmente converge en el doble de velocidad al reutilizar
valores actualizados en la misma iteración.

> En el modelo de quiroptocoria, Gauss-Seidel refleja una **retroalimentación más rápida**
> entre los factores biológicos: el tránsito digestivo (x₁) afecta inmediatamente al cálculo
> del esfuerzo de vuelo (x₂) dentro de la misma iteración.
""")

st.divider()

scenario_name = st.selectbox("Escenario:", list(SCENARIOS.keys()), key="gs_esc")
scenario = SCENARIOS[scenario_name]
A = scenario["A"].copy()
b = scenario["b"].copy()

st.markdown(f"**{scenario['desc']}**")
st.metric("κ(A)", f"{np.linalg.cond(A):.4e}")

col1, col2 = st.columns(2)
with col1:
    tol = st.select_slider("Tolerancia", [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                            value=1e-6, format_func=lambda x: f"{x:.0e}", key="gs_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", 10, 1000, 500, key="gs_max")

if st.button("▶ Ejecutar Gauss-Seidel", type="primary"):
    x_exact = solve_exact(A, b)
    x, errors, conv, info = gauss_seidel(A, b, tol=tol, max_iter=int(max_iter))

    # Reconstruir historial
    hist_x = []
    xi = np.zeros(3)
    for k in range(len(errors)):
        x_old = xi.copy()
        for i in range(3):
            s = sum(A[i, j] * xi[j] for j in range(3) if j != i)
            xi[i] = (b[i] - s) / A[i, i]
        hist_x.append(xi.copy())

    st.subheader("📋 Iteraciones")
    show_all = st.checkbox("Mostrar todas", value=False, key="gs_all")
    rows = [{"Iter": k+1, "x₁": round(xk[0],6), "x₂": round(xk[1],6),
             "x₃": round(xk[2],6), "‖error‖∞": f"{err:.2e}"}
            for k, (xk, err) in enumerate(zip(hist_x, errors))]
    df = pd.DataFrame(rows)
    n_show = len(df) if show_all else min(10, len(df))
    st.dataframe(df.head(n_show), use_container_width=True, hide_index=True)
    if not show_all and len(df) > n_show:
        st.caption(f"... {len(df)} iteraciones totales.")

    if conv:
        st.success(f"✅ Convergió en **{len(errors)} iteraciones**")
    else:
        st.error(f"❌ No convergió en {int(max_iter)} iteraciones. Error final: {errors[-1]:.2e}")

    c1, c2, c3 = st.columns(3)
    for ci, (col, nm) in enumerate(zip([c1,c2,c3],
            ["x₁ Tránsito","x₂ Vuelo","x₃ Cobertura"])):
        with col:
            st.metric(nm, f"{x[ci]:.6f}")

    if x_exact is not None:
        st.info(f"‖x_GS − x_exacta‖∞ = {np.linalg.norm(x - x_exact, np.inf):.2e}")

    # Comparación Jacobi vs GS
    from algorithms import jacobi
    x_jac, err_jac, _, _ = jacobi(A, b, tol=tol, max_iter=int(max_iter))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(err_jac)+1), err_jac,
                color=METHOD_COLORS["Jacobi"], linewidth=2, linestyle="--", label="Jacobi")
    ax.semilogy(range(1, len(errors)+1), errors,
                color=METHOD_COLORS["Gauss-Seidel"], linewidth=2, label="Gauss-Seidel")
    ax.axhline(tol, color="gray", linestyle=":", label=f"Tolerancia {tol:.0e}")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Error ‖eₖ‖∞")
    ax.set_title("Comparación: Jacobi vs Gauss-Seidel")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🗺️ Visualización 3D")
    fig3d = plot_planes_3d(A, b, title=f"Gauss-Seidel — {scenario_name}")
    if fig3d:
        st.pyplot(fig3d, use_container_width=True)
