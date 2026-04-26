import streamlit as st
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import input_system, plot_planes

st.set_page_config(page_title="Gradiente Conjugado Precondicionado", layout="wide")
st.title("🧮 Gradiente Conjugado Precondicionado (PCG)")
st.markdown("""
**Método Avanzado.** Resuelve sistemas **Ax = b** con A simétrica y definida positiva.
El precondicionador **M** (aquí se usa la diagonal de A — precondicionador de Jacobi) 
acelera la convergencia transformando el problema:

$$M^{-1}Ax = M^{-1}b$$

**Algoritmo PCG:**
1. r₀ = b − Ax₀,  z₀ = M⁻¹r₀,  p₀ = z₀
2. αₖ = (rₖᵀzₖ) / (pₖᵀApₖ)
3. xₖ₊₁ = xₖ + αₖpₖ
4. rₖ₊₁ = rₖ − αₖApₖ
5. zₖ₊₁ = M⁻¹rₖ₊₁
6. βₖ = (rₖ₊₁ᵀzₖ₊₁) / (rₖᵀzₖ)
7. pₖ₊₁ = zₖ₊₁ + βₖpₖ
""")

A, b, ok = input_system(key_prefix="pcg")

col1, col2 = st.columns(2)
with col1:
    tol = st.number_input("Tolerancia", value=1e-8, format="%.2e", key="pcg_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", value=100, min_value=5, max_value=1000, key="pcg_iter")

if ok and st.button("▶ Ejecutar PCG"):
    # Verificar simetría y definición positiva
    if not np.allclose(A, A.T):
        st.warning("⚠️ La matriz no es simétrica. PCG requiere A simétrica y definida positiva.")
    
    eigvals = np.linalg.eigvals(A)
    if not np.all(eigvals > 0):
        st.warning("⚠️ La matriz no es definida positiva. Los resultados pueden ser incorrectos.")

    x = np.zeros(3)
    r = b - A @ x
    # Precondicionador diagonal (Jacobi)
    M_inv = np.diag(1.0 / np.diag(A))
    z = M_inv @ r
    p = z.copy()
    rz_old = r @ z

    history = []
    converged = False

    for k in range(int(max_iter)):
        Ap = A @ p
        alpha = rz_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv @ r
        rz_new = r @ z
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

        res_norm = np.linalg.norm(r)
        history.append({
            "iter": k + 1,
            "x1": x[0], "x2": x[1], "x3": x[2],
            "alpha": alpha, "beta": beta,
            "‖r‖": res_norm
        })

        if res_norm < tol:
            converged = True
            break

    st.subheader("Tabla de iteraciones")
    import pandas as pd
    df = pd.DataFrame(history)
    df.columns = ["Iter", "x₁", "x₂", "x₃", "α", "β", "‖r‖"]
    st.dataframe(df.style.format({
        "x₁": "{:.6f}", "x₂": "{:.6f}", "x₃": "{:.6f}",
        "α": "{:.4f}", "β": "{:.4f}", "‖r‖": "{:.2e}"
    }), use_container_width=True)

    if converged:
        st.success(f"✅ Convergió en {len(history)} iteraciones: x₁={x[0]:.6f},  x₂={x[1]:.6f},  x₃={x[2]:.6f}")
    else:
        st.warning(f"⚠️ No convergió en {int(max_iter)} iteraciones. Residual final: {history[-1]['‖r‖']:.2e}")

    residual = np.linalg.norm(A @ x - b)
    st.info(f"Residual final ‖Ax - b‖ = {residual:.2e}")

    st.subheader("Visualización 3D — Intersección de planos")
    fig = plot_planes(A, b)
    if fig:
        st.pyplot(fig)
