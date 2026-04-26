import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, sor, gauss_seidel
from plots import plot_planes_3d, METHOD_COLORS

st.set_page_config(page_title="SOR", layout="wide")
st.title("⚡ Método Avanzado: SOR — Sobrerelajación Sucesiva")

st.markdown("""
**Método Avanzado.** Generalización de Gauss-Seidel que introduce un **parámetro de relajación ω**
para acelerar o estabilizar la convergencia:

$$x_i^{(k+1)} = (1-\\omega)\\,x_i^{(k)} + \\frac{\\omega}{a_{ii}}\\left(b_i - \\sum_{j<i} a_{ij}\\,x_j^{(k+1)} - \\sum_{j>i} a_{ij}\\,x_j^{(k)}\\right)$$

| Valor de ω | Efecto |
|---|---|
| ω = 1 | Equivale exactamente a Gauss-Seidel |
| 0 < ω < 1 | **Subrelajación** — mayor estabilidad |
| 1 < ω < 2 | **Sobrerelajación** — mayor velocidad (si ω es bien elegido) |

> **Contexto biológico:** En el escenario de migración forzada (**Estrés**), el sistema
> tiene coeficientes de gran magnitud. Con ω óptimo, SOR puede reducir drásticamente
> las iteraciones necesarias para encontrar el equilibrio metabólico.
""")

st.divider()

scenario_name = st.selectbox("Escenario:", list(SCENARIOS.keys()), key="sor_esc")
scenario = SCENARIOS[scenario_name]
A = scenario["A"].copy()
b = scenario["b"].copy()

st.markdown(f"**{scenario['desc']}**")
st.metric("κ(A)", f"{np.linalg.cond(A):.4e}")

col1, col2, col3 = st.columns(3)
with col1:
    omega = st.slider("Parámetro ω", 0.10, 1.99, 1.25, 0.01, key="sor_omega")
with col2:
    tol = st.select_slider("Tolerancia", [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                            value=1e-6, format_func=lambda x: f"{x:.0e}", key="sor_tol")
with col3:
    max_iter = st.number_input("Máx. iteraciones", 10, 1000, 500, key="sor_max")

# Explorador de ω
st.subheader("🔬 Explorador de ω — ¿Cuál es el mejor parámetro?")
st.markdown("Compara el número de iteraciones para distintos valores de ω:")

if st.button("🔍 Explorar ω óptimo"):
    omegas = np.arange(0.1, 2.0, 0.05)
    iter_counts = []
    for w in omegas:
        _, errs, conv, _ = sor(A, b, omega=float(w), tol=float(tol), max_iter=int(max_iter))
        iter_counts.append(len(errs) if conv else int(max_iter))

    fig_w, ax_w = plt.subplots(figsize=(8, 3.5))
    ax_w.plot(omegas, iter_counts, color="#e04f9a", linewidth=2)
    ax_w.axvline(x=omega, color="red", linestyle="--", label=f"ω actual = {omega}")
    best_w = omegas[np.argmin(iter_counts)]
    ax_w.axvline(x=best_w, color="green", linestyle="--", label=f"ω óptimo ≈ {best_w:.2f}")
    ax_w.set_xlabel("ω"); ax_w.set_ylabel("Iteraciones para converger")
    ax_w.set_title("Iteraciones vs ω (SOR)")
    ax_w.legend(); ax_w.grid(alpha=0.3)
    fig_w.tight_layout()
    st.pyplot(fig_w, use_container_width=True)
    st.info(f"ω óptimo encontrado: **{best_w:.2f}** con {min(iter_counts)} iteraciones.")

if st.button("▶ Ejecutar SOR", type="primary"):
    x_exact = solve_exact(A, b)
    x, errors, conv, info = sor(A, b, omega=omega, tol=float(tol), max_iter=int(max_iter))
    x_gs, errors_gs, _, _ = gauss_seidel(A, b, tol=float(tol), max_iter=int(max_iter))

    st.subheader(f"📋 Iteraciones (ω = {omega})")
    # Reconstruir historial
    hist_x = []
    xi = np.zeros(3)
    for err in errors:
        x_old = xi.copy()
        for i in range(3):
            gs = (b[i] - sum(A[i, j]*xi[j] for j in range(3) if j != i)) / A[i, i]
            xi[i] = (1 - omega)*xi[i] + omega*gs
        hist_x.append(xi.copy())

    show_all = st.checkbox("Mostrar todas", value=False, key="sor_all")
    rows = [{"Iter": k+1, "x₁": round(xk[0],6), "x₂": round(xk[1],6),
             "x₃": round(xk[2],6), "‖error‖∞": f"{err:.2e}"}
            for k, (xk, err) in enumerate(zip(hist_x, errors))]
    df = pd.DataFrame(rows)
    n_show = len(df) if show_all else min(10, len(df))
    st.dataframe(df.head(n_show), use_container_width=True, hide_index=True)

    if conv:
        st.success(f"✅ Convergió en **{len(errors)} iteraciones** (ω = {omega})")
        st.info(f"Gauss-Seidel necesitó: {len(errors_gs)} iteraciones — "
                f"{'SOR fue más rápido 🚀' if len(errors) < len(errors_gs) else 'Gauss-Seidel fue más rápido'}")
    else:
        st.error(f"❌ No convergió. Error final: {errors[-1]:.2e}")

    c1, c2, c3 = st.columns(3)
    for ci, (col, nm) in enumerate(zip([c1,c2,c3],
            ["x₁ Tránsito","x₂ Vuelo","x₃ Cobertura"])):
        with col:
            st.metric(nm, f"{x[ci]:.6f}")

    if x_exact is not None:
        st.info(f"‖x_SOR − x_exacta‖∞ = {np.linalg.norm(x - x_exact, np.inf):.2e}")

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(errors_gs)+1), errors_gs,
                color=METHOD_COLORS["Gauss-Seidel"], linewidth=2,
                linestyle="--", label="Gauss-Seidel")
    ax.semilogy(range(1, len(errors)+1), errors,
                color=METHOD_COLORS["SOR"], linewidth=2, label=f"SOR (ω={omega})")
    ax.axhline(float(tol), color="gray", linestyle=":", label=f"Tol {float(tol):.0e}")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Error ‖eₖ‖∞")
    ax.set_title("SOR vs Gauss-Seidel")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🗺️ Visualización 3D")
    fig3d = plot_planes_3d(A, b, title=f"SOR (ω={omega}) — {scenario_name}")
    if fig3d:
        st.pyplot(fig3d, use_container_width=True)
