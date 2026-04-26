import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Título y descripción
st.set_page_config(page_title="Métodos Iterativos", layout="wide")
st.title("🧮 Solucionador de Sistemas: Método de Jacobi")
st.write("Ingresa los valores de la matriz **A** y el vector **b** para resolver $Ax = b$.")

# --- INTERFAZ DE ENTRADA ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Matriz de Coeficientes (A)")
    # Creamos una cuadrícula de 3x3 usando columnas de Streamlit
    A = np.zeros((3, 3))
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            A[i, j] = cols[j].number_input(f"A[{i+1},{j+1}]", value=float(i==j), key=f"A{i}{j}")

with col2:
    st.subheader("Vector (b)")
    b = np.zeros(3)
    for i in range(3):
        b[i] = st.number_input(f"b[{i+1}]", value=1.0, key=f"b{i}")

# Parámetros del método en el lateral
st.sidebar.header("Configuración")
tol = st.sidebar.number_input("Tolerancia", value=1e-5, format="%.5e")
max_iter = st.sidebar.slider("Máximo de iteraciones", 1, 100, 50)

# --- LÓGICA DEL MÉTODO DE JACOBI ---
def jacobi(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    errores = []
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - suma) / A[i, i]
        
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errores.append(error)
        x = x_new
        
        if error < tol:
            return x, errores, True
    return x, errores, False

# --- BOTÓN Y RESULTADOS ---
if st.button("Resolver Sistema"):
    # Verificar diagonal dominante (opcional pero recomendado)
    if any(abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(3) if i != j) for i in range(3)):
        st.warning("⚠️ La matriz no es estrictamente diagonal dominante. El método podría no converger.")

    solucion, historial_error, convergio = jacobi(A, b, tol, max_iter)

    if convergio:
        st.success(f"✅ ¡Convergió en {len(historial_error)} iteraciones!")
    else:
        st.error("❌ No se alcanzó la tolerancia en el máximo de iteraciones.")

    # Mostrar resultados
    res_cols = st.columns(2)
    res_cols[0].metric("x1", round(solucion[0], 4))
    res_cols[1].metric("x2", round(solucion[1], 4))
    st.metric("x3", round(solucion[2], 4))

    # Gráfica de convergencia
    fig, ax = plt.subplots()
    ax.plot(historial_error, marker='o', color='purple')
    ax.set_yscale('log') # Escala logarítmica para ver mejor el error
    ax.set_title("Convergencia del Error")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error (Norma Inf)")
    ax.grid(True)
    st.pyplot(fig)