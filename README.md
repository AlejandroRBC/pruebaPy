# 🦇 Quiroptocoria — Métodos Numéricos

Aplicación interactiva en Streamlit para el análisis del modelo de **Dinámica de Reforestación
por Dispersión de Semillas por Murciélagos (Quiroptocoria)**.

Proyecto de Métodos Numéricos | UMSA | Abril 2026

## Estructura del proyecto

```
app.py                          ← Página principal: contexto, variables, solución exacta, editor
pages/
  1_Factorizacion_LU.py         ← Método directo con paso a paso
  2_Jacobi.py                   ← Método iterativo clásico
  3_Gauss_Seidel.py             ← Método iterativo clásico
  4_SOR.py                      ← Sobrerelajación Sucesiva + explorador de ω
  5_Gradiente_Conjugado_PCG.py  ← PCG basado en Suñagua (2020)
  6_Analisis_Comparativo.py     ← Tabla comparativa automática
algorithms.py                   ← Todos los algoritmos numéricos
plots.py                        ← Funciones de visualización
requirements.txt
```

## Escenarios modelados

| Escenario | Descripción |
|---|---|
| 🌿 Bosque Saludable | Caso ideal — matriz diagonal dominante |
| 🔥 Fragmentación del Hábitat | Caso estrés — coeficientes de alta magnitud |
| 💀 Deforestación Severa | Caso mal condicionado — hiperplanos casi paralelos |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Conecta el repo y apunta a `app.py`.

## Referencia metodológica

> Suñagua, P. (2020). *Método de Gradientes Conjugados Precondicionado.*
> Revista Boliviana de Matemática, UMSA 04, 2–7.
