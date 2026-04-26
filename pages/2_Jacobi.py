import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, jacobi
from plots import plot_planes_3d, plot_convergence, METHOD_COLORS

st.set_page_config(page_title="Método de Jacobi", layout="wide")
st.title("🔄 Método Iterativo: Jacobi")

st.markdown("""
**Método Iterativo Clásico.** En cada iteración actualiza **todas las variables
simultáneamente** usando los valores de la iteración anterior:

$$x_i^{(k+1)} = \\frac{1}{a_{ii}}\\left(b_i - \\sum_{j \\neq i} a_{ij}\\, x_j^{(k)}\\right)$$

**Condición de convergencia:** La matriz debe ser **diagonal dominante estricta**, es decir,
$|a_{ii}| > \\sum_{j \\neq i} |a_{ij}|$ para toda fila $i$.

> En el modelo de quiroptocoria, el **Bosque Saludable** cumple esta condición. El escenario
> de **Deforestación** viola la convergencia por el alto número de condición.
""")

st.divider()

scenario_name = st.selectbox("Escenario:", list(SCENARIOS.keys()), key="jac_esc")
scenario = SCENARIOS[scenario_name]
A = scenario["A"].copy()
b = scenario["b"].copy()

st.markdown(f"**{scenario['desc']}**")

# Verificación diagonal dominante
is_dd = all(abs(A[i, i]) > sum(abs(A[i, j]) for j in range(3) if j != i) for i in range(3))
if is_dd:
    st.success("✅ La matriz es diagonal dominante — convergencia garantizada para Jacobi.")
else:
    st.warning("⚠️ La matriz NO es diagonal dominante — Jacobi puede no converger.")

col1, col2 = st.columns(2)
with col1:
    tol = st.select_slider("Tolerancia", [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                            value=1e-6, format_func=lambda x: f"{x:.0e}", key="jac_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", 10, 1000, 500, key="jac_max")

if st.button("▶ Ejecutar Jacobi", type="primary"):
    x_exact = solve_exact(A, b)
    x, errors, conv, info = jacobi(A, b, tol=tol, max_iter=int(max_iter))

    # Tabla de iteraciones (primeras + últimas)
    st.subheader("📋 Iteraciones")
    show_all = st.checkbox("Mostrar todas las iteraciones", value=False, key="jac_all")

    # Reconstruir historial
    hist_x = [np.zeros(3)]
    xi = np.zeros(3)
    for k in range(len(errors)):
        xi_new = np.zeros(3)
        for i in range(3):
            s = sum(A[i, j] * xi[j] for j in range(3) if j != i)
            xi_new[i] = (b[i] - s) / A[i, i]
        hist_x.append(xi_new.copy())
        xi = xi_new.copy()

    rows = []
    for k, (xk, err) in enumerate(zip(hist_x[1:], errors)):
        rows.append({"Iter": k+1, "x₁": round(xk[0],6),
                     "x₂": round(xk[1],6), "x₃": round(xk[2],6),
                     "‖error‖∞": f"{err:.2e}"})

    df = pd.DataFrame(rows)
    if show_all:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        n_show = min(10, len(df))
        st.dataframe(df.head(n_show), use_container_width=True, hide_index=True)
        if len(df) > n_show:
            st.caption(f"... mostrando las primeras {n_show} de {len(df)} iteraciones.")

    if conv:
        st.success(f"✅ Convergió en **{len(errors)} iteraciones** — tolerancia {tol:.0e} alcanzada.")
    else:
        st.error(f"❌ No convergió en {int(max_iter)} iteraciones. Error final: {errors[-1]:.2e}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    for ci, (col, nm) in enumerate(zip([c1,c2,c3],
            ["x₁ Tránsito","x₂ Vuelo","x₃ Cobertura"])):
        with col:
            st.metric(nm, f"{x[ci]:.6f}")

    if x_exact is not None:
        st.info(f"‖x_Jacobi − x_exacta‖∞ = {np.linalg.norm(x - x_exact, np.inf):.2e}")

    # Gráfico de convergencia
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(errors)+1), errors,
                color=METHOD_COLORS["Jacobi"], linewidth=2, label="Jacobi")
    ax.axhline(tol, color="gray", linestyle="--", label=f"Tolerancia {tol:.0e}")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Error ‖eₖ‖∞")
    ax.set_title("Convergencia Jacobi"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🗺️ Visualización 3D")
    fig3d = plot_planes_3d(A, b, title=f"Jacobi — {scenario_name}")
    if fig3d:
        st.pyplot(fig3d, use_container_width=True)
