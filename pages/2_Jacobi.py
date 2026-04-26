import streamlit as st
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import input_system, plot_planes

st.set_page_config(page_title="Método de Jacobi", layout="wide")
st.title("🔄 Método de Jacobi")
st.markdown("""
**Método Iterativo.** A partir de una aproximación inicial x⁽⁰⁾, en cada iteración actualiza
**todas las variables simultáneamente** usando los valores de la iteración anterior:

$$x_i^{(k+1)} = \\frac{1}{a_{ii}}\\left(b_i - \\sum_{j \\neq i} a_{ij} x_j^{(k)}\\right)$$

Requiere que la matriz sea **diagonal dominante** para garantizar convergencia.
""")

A, b, ok = input_system(key_prefix="jacobi")

col1, col2, col3 = st.columns(3)
with col1:
    tol = st.number_input("Tolerancia", value=1e-6, format="%.2e", key="jac_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", value=50, min_value=5, max_value=500, key="jac_iter")
with col3:
    x0_str = st.text_input("x inicial (separado por espacios)", value="0 0 0", key="jac_x0")

if ok and st.button("▶ Ejecutar Jacobi"):
    try:
        x = np.array(list(map(float, x0_str.split())))
    except:
        st.error("Formato de x inicial inválido.")
        st.stop()

    n = 3
    history = []
    converged = False

    for k in range(int(max_iter)):
        x_new = np.zeros(n)
        for i in range(n):
            if abs(A[i, i]) < 1e-12:
                st.error(f"Elemento diagonal nulo en fila {i+1}.")
                st.stop()
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        err = np.linalg.norm(x_new - x, np.inf)
        history.append({"iter": k + 1, "x1": x_new[0], "x2": x_new[1], "x3": x_new[2], "error": err})
        x = x_new.copy()

        if err < tol:
            converged = True
            break

    st.subheader("Tabla de iteraciones")
    import pandas as pd
    df = pd.DataFrame(history)
    df.columns = ["Iter", "x₁", "x₂", "x₃", "‖error‖∞"]
    st.dataframe(df.style.format({
        "x₁": "{:.6f}", "x₂": "{:.6f}", "x₃": "{:.6f}", "‖error‖∞": "{:.2e}"
    }), use_container_width=True)

    if converged:
        st.success(f"✅ Convergió en {len(history)} iteraciones: x₁={x[0]:.6f},  x₂={x[1]:.6f},  x₃={x[2]:.6f}")
    else:
        st.warning(f"⚠️ No convergió en {int(max_iter)} iteraciones. Último error: {history[-1]['error']:.2e}")

    st.subheader("Visualización 3D — Intersección de planos")
    fig = plot_planes(A, b)
    if fig:
        st.pyplot(fig)
