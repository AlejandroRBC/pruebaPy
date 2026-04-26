import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(__file__))
from algorithms import SCENARIOS, VAR_NAMES, solve_exact, run_all_methods, lu_factorization
from plots import plot_planes_3d, plot_convergence

st.set_page_config(
    page_title="Quiroptocoria — Métodos Numéricos",
    page_icon="🦇",
    layout="wide",
)

# ─── Encabezado ──────────────────────────────────────────────────────────────
st.title("🦇 Modelado de la Quiroptocoria")
st.subheader("Dinámica de Reforestación por Dispersión de Semillas por Murciélagos")
st.markdown("*Proyecto — Métodos Numéricos | Abril 2026*")
st.divider()

# ─── Contexto del problema ───────────────────────────────────────────────────
with st.expander("📖 Contexto y Justificación del Problema", expanded=True):
    st.markdown("""
Los **murciélagos frugívoros** son agentes fundamentales en la regeneración de selvas
tropicales. Al alimentarse, transportan semillas en su tracto digestivo y las dispersan
durante el vuelo. El éxito de la germinación depende del equilibrio entre tres factores
biológicos que modelamos como un sistema lineal **Ax = b**.

> **Referencia metodológica:** Los métodos avanzados siguen el Algoritmo Mz de
> *Suñagua, P. (2020) — Método de Gradientes Conjugados Precondicionado.*
> Revista Boliviana de Matemática, UMSA.
""")

# ─── Definición de variables ─────────────────────────────────────────────────
st.subheader("📐 Definición de Variables")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
#### Variables de decisión (xₙ)

| Variable | Significado biológico |
|---|---|
| **x₁** | Tasa de tránsito digestivo — velocidad de procesamiento metabólico de las semillas |
| **x₂** | Esfuerzo metabólico del vuelo — energía consumida por kilómetro recorrido |
| **x₃** | Índice de cobertura vegetal — protección y disponibilidad de recursos del suelo |
""")

with col2:
    st.markdown("""
#### Términos independientes (b)

| Término | Significado biológico |
|---|---|
| **b₁** | Nivel de biodiversidad deseado en el ecosistema |
| **b₂** | Tasa de regeneración mínima necesaria para contrarrestar la erosión |
| **b₃** | Capacidad de carga de nutrientes del suelo |
""")

st.info("""
**Interpretación del sistema Ax = b:** Cada ecuación impone un balance de recursos
en el ecosistema. Los coeficientes aᵢⱼ representan cómo cada variable biológica
(x₁, x₂, x₃) contribuye al cumplimiento de cada requerimiento ecosistémico (b₁, b₂, b₃).
""")

st.divider()

# ─── Selector de escenario ───────────────────────────────────────────────────
st.subheader("⚙️ Selector de Escenario")

scenario_name = st.selectbox(
    "Elige el escenario de análisis:",
    list(SCENARIOS.keys()),
    key="scenario_main",
)
scenario = SCENARIOS[scenario_name]
A_default = scenario["A"].copy()
b_default = scenario["b"].copy()

st.markdown(f"**Descripción:** {scenario['desc']}")

# Número de condición
cond = np.linalg.cond(A_default)
col_c1, col_c2 = st.columns(2)
with col_c1:
    st.metric("Número de condición κ(A)", f"{cond:.4e}")
with col_c2:
    st.metric("Estado", scenario["cond_label"])

st.divider()

# ─── Editor de matriz dinámico ───────────────────────────────────────────────
st.subheader("✏️ Editor de Matriz Dinámica")
st.markdown("Modifica cualquier coeficiente aᵢⱼ o término bᵢ y la solución se recalcula al instante.")

col_ed, col_b = st.columns([3, 1])

A_edit = np.zeros((3, 3))
with col_ed:
    st.markdown("**Matriz A**")
    cols_a = st.columns(3)
    labels_col = ["x₁", "x₂", "x₃"]
    for i in range(3):
        for j in range(3):
            with cols_a[j]:
                A_edit[i, j] = st.number_input(
                    f"a[{i+1},{j+1}]",
                    value=float(A_default[i, j]),
                    format="%.4f",
                    key=f"a_{i}_{j}",
                    step=0.001,
                )

b_edit = np.zeros(3)
with col_b:
    st.markdown("**Vector b**")
    for i in range(3):
        b_edit[i] = st.number_input(
            f"b[{i+1}]",
            value=float(b_default[i]),
            format="%.4f",
            key=f"b_{i}",
            step=0.001,
        )

# ─── Solución exacta ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🎯 Solución Exacta Teórica")

x_exact = solve_exact(A_edit, b_edit)

if x_exact is not None:
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("x₁ — Tránsito digestivo", f"{x_exact[0]:.8f}")
    with col_s2:
        st.metric("x₂ — Esfuerzo de vuelo", f"{x_exact[1]:.8f}")
    with col_s3:
        st.metric("x₃ — Cobertura vegetal", f"{x_exact[2]:.8f}")

    residual = np.linalg.norm(A_edit @ x_exact - b_edit)
    st.markdown(f"**Residual ‖Ax − b‖₂ = `{residual:.2e}`** — obtenido por eliminación gaussiana con pivoteo.")

    # Interpretación
    with st.expander("🔬 Interpretación biológica de la solución"):
        st.markdown(f"""
La solución x = ({x_exact[0]:.4f}, {x_exact[1]:.4f}, {x_exact[2]:.4f}) representa el
**punto de equilibrio del ecosistema** para el escenario seleccionado:

- **x₁ = {x_exact[0]:.4f}:** Tasa de tránsito digestivo óptima. Un valor {'alto' if x_exact[0] > 1 else 'bajo'} indica
  {'mayor velocidad metabólica, asociada a alta densidad de semillas disponibles.' if x_exact[0] > 1 else 'procesamiento lento, propio de ecosistemas con baja disponibilidad de frutos.'}

- **x₂ = {x_exact[1]:.4f}:** Esfuerzo de vuelo requerido. Refleja la {'alta' if x_exact[1] > 1 else 'baja'} demanda
  energética para mantener la dispersión efectiva de semillas.

- **x₃ = {x_exact[2]:.4f}:** Índice de cobertura vegetal necesario para asegurar la
  germinación exitosa según los requerimientos del ecosistema.
""")
else:
    st.error("⚠️ El sistema no tiene solución única (matriz singular). Verifica los coeficientes.")

st.divider()

# ─── Gráfico 3D de planos ─────────────────────────────────────────────────────
st.subheader("🗺️ Visualización 3D — Intersección de Planos")

if x_exact is not None:
    fig_3d = plot_planes_3d(A_edit, b_edit, title=f"Planos — {scenario_name}")
    if fig_3d:
        st.pyplot(fig_3d, use_container_width=True)

    if "Mal Condicionado" in scenario_name or "Deforestación" in scenario_name:
        st.warning("""
⚠️ **Caso Mal Condicionado:** Observa cómo los planos de las ecuaciones 1 y 2 son
casi paralelos (filas casi proporcionales). Esto genera un número de condición κ(A) 
extremadamente alto, causando inestabilidad numérica en los métodos iterativos clásicos.
La intersección existe matemáticamente, pero es numéricamente muy sensible a perturbaciones.
""")
else:
    st.info("No se puede graficar: el sistema es singular.")

st.divider()

# ─── Panel comparativo rápido ─────────────────────────────────────────────────
st.subheader("📊 Panel Comparativo Rápido")
st.markdown("Ejecuta todos los métodos con los valores actuales de la matriz.")

col_omega, col_tol = st.columns(2)
with col_omega:
    omega_val = st.slider("Parámetro ω (SOR)", 0.10, 1.99, 1.25, 0.01, key="omega_main")
with col_tol:
    tol_val = st.select_slider(
        "Tolerancia",
        options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
        key="tol_main",
    )

if st.button("▶ Ejecutar todos los métodos", type="primary"):
    if x_exact is None:
        st.error("Sistema singular — no se puede resolver.")
    else:
        with st.spinner("Calculando..."):
            results = run_all_methods(A_edit, b_edit, omega=omega_val, tol=tol_val)

        # Tabla comparativa
        st.markdown("#### Tabla Comparativa de Eficiencia")
        rows = []
        for method, data in results.items():
            rows.append({
                "Método": method,
                "Iter. necesarias": data["iters"],
                "Convergió": "✅ Sí" if data["conv"] else "❌ No",
                "‖x − x*‖∞": f"{np.linalg.norm(data['x'] - x_exact, np.inf):.2e}" if data["x"] is not None else "—",
            })

        # LU (no iterativo)
        L, U, x_lu, _ = lu_factorization(A_edit, b_edit)
        if x_lu is not None:
            rows.append({
                "Método": "Factorización LU",
                "Iter. necesarias": "— (directo)",
                "Convergió": "✅ Sí",
                "‖x − x*‖∞": f"{np.linalg.norm(x_lu - x_exact, np.inf):.2e}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Gráfico de convergencia
        st.markdown("#### Gráfica de Convergencia — Error vs. Iteración")
        fig_conv = plot_convergence(results, tol=tol_val)
        st.pyplot(fig_conv, use_container_width=True)

        # Soluciones individuales
        with st.expander("Ver soluciones individuales por método"):
            for method, data in results.items():
                if data["x"] is not None:
                    x = data["x"]
                    st.markdown(f"**{method}:** x₁={x[0]:.6f}, x₂={x[1]:.6f}, x₃={x[2]:.6f}")

st.divider()
st.caption("Métodos Numéricos · UMSA · Abril 2026 · Tema: Quiroptocoria")
