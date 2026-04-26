import streamlit as st

st.set_page_config(
    page_title="Métodos Numéricos",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Métodos Numéricos — Sistemas de Ecuaciones Lineales")
st.markdown("---")

st.markdown("""
Selecciona un método en el menú lateral para ver el algoritmo paso a paso.

### Métodos disponibles

| Categoría | Método |
|---|---|
| Directo | Factorización LU |
| Iterativo clásico | Jacobi |
| Iterativo clásico | Gauss-Seidel |
| Avanzado | SOR (Sobrerelajación Sucesiva) |
| Avanzado | Gradiente Conjugado Precondicionado |

> **Todos los ejemplos usan sistemas 3×3** con visualización 3D del plano de solución.
""")

st.info("👈 Usa el menú lateral para navegar entre los métodos.")
