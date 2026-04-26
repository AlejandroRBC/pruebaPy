import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms import SCENARIOS, solve_exact, jacobi, gauss_seidel, sor, pcg, lu_factorization
from plots import plot_convergence, METHOD_COLORS

st.set_page_config(page_title="Análisis Comparativo", layout="wide")
st.title("📊 Análisis Comparativo de Eficiencia")
st.markdown("""
Comparación del desempeño de los cinco métodos en los **tres escenarios** del modelo
de quiroptocoria. Se llena automáticamente la tabla del enunciado (tolerancia 10⁻⁶).
""")

st.divider()

tol = st.select_slider("Tolerancia global", [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],
                        value=1e-6, format_func=lambda x: f"{x:.0e}", key="comp_tol")
omega = st.slider("ω para SOR", 0.10, 1.99, 1.25, 0.01, key="comp_omega")
max_iter = 500

if st.button("▶ Generar Tabla Comparativa Completa", type="primary"):
    all_rows = []
    scenario_results = {}

    for sc_name, sc in SCENARIOS.items():
        A = sc["A"].copy()
        b = sc["b"].copy()
        short = sc_name.split("(")[0].strip()

        x_exact = solve_exact(A, b)

        _, err_jac,  conv_jac,  _ = jacobi(A, b, tol=float(tol), max_iter=max_iter)
        _, err_gs,   conv_gs,   _ = gauss_seidel(A, b, tol=float(tol), max_iter=max_iter)
        _, err_sor,  conv_sor,  _ = sor(A, b, omega=omega, tol=float(tol), max_iter=max_iter)
        _, err_pcg,  conv_pcg,  _ = pcg(A, b, tol=float(tol), max_iter=max_iter)
        _, _, x_lu, _             = lu_factorization(A, b)

        scenario_results[sc_name] = {
            "Jacobi":        {"errors": err_jac,  "conv": conv_jac},
            "Gauss-Seidel":  {"errors": err_gs,   "conv": conv_gs},
            "SOR":           {"errors": err_sor,  "conv": conv_sor},
            "PCG (Suñagua)": {"errors": err_pcg,  "conv": conv_pcg},
        }

        row = {
            "Escenario": short,
            "κ(A)": f"{np.linalg.cond(A):.2e}",
            "Jacobi": f"{len(err_jac)}" if conv_jac else f">{max_iter} ❌",
            "Gauss-Seidel": f"{len(err_gs)}" if conv_gs else f">{max_iter} ❌",
            f"SOR (ω={omega})": f"{len(err_sor)}" if conv_sor else f">{max_iter} ❌",
            "PCG (Suñagua)": f"{len(err_pcg)}" if conv_pcg else f">{max_iter} ❌",
            "LU (directo)": "✅ exacto",
        }
        all_rows.append(row)

    df_comp = pd.DataFrame(all_rows)
    st.subheader("Tabla Comparativa de Iteraciones (Tolerancia 10⁻⁶)")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.download_button(
        "💾 Descargar tabla CSV",
        data=df_comp.to_csv(index=False).encode(),
        file_name="tabla_comparativa.csv",
        mime="text/csv",
    )

    st.divider()

    # Gráficos de convergencia por escenario
    st.subheader("Gráficas de Convergencia por Escenario")

    for sc_name, methods in scenario_results.items():
        st.markdown(f"#### {sc_name}")
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for method, data in methods.items():
            errs = data["errors"]
            if errs:
                color = METHOD_COLORS.get(method, "#aaa")
                conv_label = f"{len(errs)} iter." if data["conv"] else f">{max_iter} ❌"
                ax.semilogy(range(1, len(errs)+1), errs,
                            color=color, linewidth=2, label=f"{method} ({conv_label})")
        ax.axhline(float(tol), color="gray", linestyle="--", label=f"Tol {float(tol):.0e}")
        ax.set_xlabel("Iteración"); ax.set_ylabel("Error ‖eₖ‖∞")
        ax.set_title(sc_name); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("🔬 Interpretación de Resultados")

    st.markdown("""
#### Caso Ideal — Bosque Saludable
La matriz diagonal dominante garantiza convergencia en **pocos ciclos** para todos
los métodos. Gauss-Seidel mejora a Jacobi al reutilizar valores actualizados;
SOR con ω óptimo es aún más rápido. PCG converge en muy pocas iteraciones
dado el bajo κ(A).

#### Caso Estrés — Fragmentación del Hábitat
Los coeficientes de mayor magnitud aumentan el número de condición moderadamente.
Los métodos iterativos requieren más iteraciones, pero siguen convergiendo.
SOR puede acelerar la convergencia significativamente si ω se elige bien.

#### Caso Mal Condicionado — Deforestación
Con κ(A) ≫ 1 y filas casi proporcionales (hiperplanos casi paralelos), los métodos
de Jacobi y Gauss-Seidel se estancan o divergen. El **PCG precondicionado** (Suñagua,
2020) mantiene la convergencia al reducir el número de condición efectivo mediante
el precondicionador diagonal. LU proporciona la solución exacta independientemente
del condicionamiento, pero es sensible a errores de redondeo con κ muy alto.
""")
