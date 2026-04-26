import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, pcg, jacobi, gauss_seidel, sor
from plots import plot_planes_3d, METHOD_COLORS

st.set_page_config(page_title="Gradiente Conjugado Precondicionado", layout="wide")
st.title("🧮 Método Avanzado: Gradiente Conjugado Precondicionado (PCG)")

st.markdown("""
**Método Avanzado — basado en Suñagua, P. (2020).**
Resuelve sistemas **Ax = b** con A simétrica y definida positiva, mejorando el método clásico
de gradientes conjugados cuando la matriz está **mal condicionada**.
""")

with st.expander("📖 Fundamento teórico (Suñagua, 2020)", expanded=False):
    st.markdown("""
#### Idea central

Dado que la tasa de convergencia depende del número de condición κ(A), cuando la matriz
está mal condicionada se introduce una **matriz precondicionadora C** no singular tal que
la nueva matriz $M = C^{-1}AC^{-T}$ tenga autovalores concentrados cerca de 1:

$$\\kappa(M) \\ll \\kappa(A)$$

#### Precondicionador utilizado: diagonal de Jacobi

$$C = \\text{diag}(A)^{-1/2} \\quad \\Rightarrow \\quad M = \\text{diag}(A)^{-1} \\cdot A$$

Este precondicionador trivial es computacionalmente barato y mejora el condicionamiento
para matrices con diagonal dominante.

#### Algoritmo Mz (Suñagua, Algoritmo 2)

```
Datos: x₀ tal que Ax₀ ≈ b
k = 0
Calcule el residuo inicial r₀ = b − Ax₀
Mientras rₖ ≠ 0:
    Resuelva Mzₖ = rₖ          ← Aplicar precondicionador
    si k=1: p₁ = z₀
    sino:
        βₖ = (rᵀₖ₋₁ zₖ₋₁) / (rᵀₖ₋₂ zₖ₋₂)
        pₖ = zₖ₋₁ + βₖ pₖ₋₁
    αₖ = (rᵀₖ₋₁ zₖ₋₁) / (pᵀₖ Apₖ)
    xₖ = xₖ₋₁ + αₖ pₖ
    rₖ = rₖ₋₁ − αₖ Apₖ
retorne x = xₖ
```
""")

st.divider()

scenario_name = st.selectbox("Escenario:", list(SCENARIOS.keys()), key="pcg_esc")
scenario = SCENARIOS[scenario_name]
A = scenario["A"].copy()
b = scenario["b"].copy()

st.markdown(f"**{scenario['desc']}**")

cond_A = np.linalg.cond(A)
diag_inv = 1.0 / np.diag(A)
M_precond = np.diag(diag_inv) @ A
cond_M = np.linalg.cond(M_precond)

col_k1, col_k2, col_k3 = st.columns(3)
with col_k1:
    st.metric("κ(A) original", f"{cond_A:.4e}")
with col_k2:
    st.metric("κ(M) precondicionado", f"{cond_M:.4e}")
with col_k3:
    mejora = cond_A / cond_M if cond_M > 0 else 0
    st.metric("Mejora del condicionamiento", f"{mejora:.2f}×")

# Verificar simetría y DP
is_sym = np.allclose(A, A.T, atol=1e-6)
eigvals = np.linalg.eigvals(A)
is_pd = bool(np.all(eigvals > 0))

col_v1, col_v2 = st.columns(2)
with col_v1:
    if is_sym:
        st.success("✅ Matriz simétrica")
    else:
        st.warning("⚠️ Matriz no simétrica — PCG puede ser impreciso")
with col_v2:
    if is_pd:
        st.success("✅ Matriz definida positiva")
    else:
        st.warning("⚠️ Matriz no definida positiva")

col1, col2 = st.columns(2)
with col1:
    tol = st.select_slider("Tolerancia", [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                            value=1e-6, format_func=lambda x: f"{x:.0e}", key="pcg_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", 10, 1000, 500, key="pcg_max")

if st.button("▶ Ejecutar PCG (Suñagua Algoritmo 2)", type="primary"):
    x_exact = solve_exact(A, b)

    # PCG con registro de α y β
    n = len(b)
    x_pcg = np.zeros(n)
    r = b - A @ x_pcg
    z = diag_inv * r
    p = z.copy()
    rz_old = r @ z
    errors_pcg = []
    pcg_steps = []
    conv = False

    for k in range(int(max_iter)):
        Ap = A @ p
        denom = p @ Ap
        if abs(denom) < 1e-30:
            break
        alpha = rz_old / denom
        x_pcg = x_pcg + alpha * p
        r = r - alpha * Ap
        z = diag_inv * r
        rz_new = r @ z
        err = np.linalg.norm(r)
        beta = rz_new / rz_old if k > 0 else 0.0
        p = z + beta * p
        rz_old = rz_new
        errors_pcg.append(err)
        pcg_steps.append({
            "Iter": k+1, "α": round(alpha,6), "β": round(beta,6),
            "x₁": round(x_pcg[0],6), "x₂": round(x_pcg[1],6),
            "x₃": round(x_pcg[2],6), "‖r‖₂": f"{err:.2e}"
        })
        if err < float(tol):
            conv = True
            break

    # Tabla de iteraciones con α y β
    st.subheader("📋 Iteraciones con parámetros α y β")
    show_all = st.checkbox("Mostrar todas", value=False, key="pcg_all")
    df = pd.DataFrame(pcg_steps)
    n_show = len(df) if show_all else min(10, len(df))
    st.dataframe(df.head(n_show), use_container_width=True, hide_index=True)

    if conv:
        st.success(f"✅ Convergió en **{len(errors_pcg)} iteraciones** (residual ‖r‖₂ < {float(tol):.0e})")
    else:
        st.error(f"❌ No convergió. Residual final: {errors_pcg[-1]:.2e}")

    c1, c2, c3 = st.columns(3)
    for ci, (col, nm) in enumerate(zip([c1,c2,c3],
            ["x₁ Tránsito","x₂ Vuelo","x₃ Cobertura"])):
        with col:
            st.metric(nm, f"{x_pcg[ci]:.8f}")

    if x_exact is not None:
        st.info(f"‖x_PCG − x_exacta‖∞ = {np.linalg.norm(x_pcg - x_exact, np.inf):.2e}")

    st.divider()

    # Comparación completa de todos los métodos
    st.subheader("📊 Comparación con todos los métodos iterativos")
    omega_comp = 1.25
    x_jac, err_jac, conv_jac, _ = jacobi(A, b, tol=float(tol), max_iter=int(max_iter))
    x_gs,  err_gs,  conv_gs,  _ = gauss_seidel(A, b, tol=float(tol), max_iter=int(max_iter))
    x_sor, err_sor, conv_sor, _ = sor(A, b, omega=omega_comp, tol=float(tol), max_iter=int(max_iter))

    # Tabla comparativa
    comp_rows = [
        {"Método": "Jacobi",         "Iteraciones": len(err_jac), "Convergió": "✅" if conv_jac else "❌",
         "‖x − x*‖∞": f"{np.linalg.norm(x_jac - x_exact, np.inf):.2e}" if x_exact is not None else "—"},
        {"Método": "Gauss-Seidel",   "Iteraciones": len(err_gs),  "Convergió": "✅" if conv_gs  else "❌",
         "‖x − x*‖∞": f"{np.linalg.norm(x_gs  - x_exact, np.inf):.2e}" if x_exact is not None else "—"},
        {"Método": f"SOR (ω={omega_comp})", "Iteraciones": len(err_sor), "Convergió": "✅" if conv_sor else "❌",
         "‖x − x*‖∞": f"{np.linalg.norm(x_sor - x_exact, np.inf):.2e}" if x_exact is not None else "—"},
        {"Método": "PCG (Suñagua)",  "Iteraciones": len(errors_pcg), "Convergió": "✅" if conv else "❌",
         "‖x − x*‖∞": f"{np.linalg.norm(x_pcg - x_exact, np.inf):.2e}" if x_exact is not None else "—"},
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Gráfico de convergencia comparativo
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, errs, color in [
        ("Jacobi",       err_jac,    METHOD_COLORS["Jacobi"]),
        ("Gauss-Seidel", err_gs,     METHOD_COLORS["Gauss-Seidel"]),
        ("SOR",          err_sor,    METHOD_COLORS["SOR"]),
        ("PCG (Suñagua)",errors_pcg, METHOD_COLORS["PCG (Suñagua)"]),
    ]:
        if errs:
            ax.semilogy(range(1, len(errs)+1), errs, color=color,
                        linewidth=2, label=f"{name} ({len(errs)} iter.)")
    ax.axhline(float(tol), color="gray", linestyle="--", label=f"Tol {float(tol):.0e}")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Error / Residual (log)")
    ax.set_title(f"Convergencia comparativa — {scenario_name}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🗺️ Visualización 3D")
    fig3d = plot_planes_3d(A, b, title=f"PCG — {scenario_name}")
    if fig3d:
        st.pyplot(fig3d, use_container_width=True)

    if "Deforestación" in scenario_name or "Mal Condicionado" in scenario_name:
        st.warning("""
**Caso Mal Condicionado — Nota de Suñagua (2020):**
El error de acarreo por redondeo puede ser significativo cuando κ(A) es muy alto.
El precondicionador diagonal reduce κ(A), pero para matrices muy mal condicionadas
se recomienda la factorización incompleta de Cholesky como precondicionador.
""")
