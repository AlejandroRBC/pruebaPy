import streamlit as st
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import input_system, plot_planes

st.set_page_config(page_title="Gauss-Seidel", layout="wide")
st.title("🔁 Método de Gauss-Seidel")
st.markdown("""
**Método Iterativo.** Similar a Jacobi, pero actualiza cada variable **inmediatamente**
usando los valores más recientes ya calculados en la misma iteración:

$$x_i^{(k+1)} = \\frac{1}{a_{ii}}\\left(b_i - \\sum_{j < i} a_{ij} x_j^{(k+1)} - \\sum_{j > i} a_{ij} x_j^{(k)}\\right)$$

Generalmente converge más rápido que Jacobi.
""")

A, b, ok = input_system(key_prefix="gs")

col1, col2, col3 = st.columns(3)
with col1:
    tol = st.number_input("Tolerancia", value=1e-6, format="%.2e", key="gs_tol")
with col2:
    max_iter = st.number_input("Máx. iteraciones", value=50, min_value=5, max_value=500, key="gs_iter")
with col3:
    x0_str = st.text_input("x inicial (separado por espacios)", value="0 0 0", key="gs_x0")

if ok and st.button("▶ Ejecutar Gauss-Seidel"):
    try:
        x = np.array(list(map(float, x0_str.split())))
    except:
        st.error("Formato de x inicial inválido.")
        st.stop()

    n = 3
    history = []
    converged = False

    for k in range(int(max_iter)):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-12:
                st.error(f"Elemento diagonal nulo en fila {i+1}.")
                st.stop()
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i, i]

        err = np.linalg.norm(x - x_old, np.inf)
        history.append({"iter": k + 1, "x1": x[0], "x2": x[1], "x3": x[2], "error": err})

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
