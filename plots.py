import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

METHOD_COLORS = {
    "Jacobi":        "#e07b39",
    "Gauss-Seidel":  "#4e9af1",
    "SOR":           "#6ef14e",
    "PCG (Suñagua)": "#e04f9a",
}

def plot_planes_3d(A, b, title="Intersección de planos — Sistema Ax = b"):
    """Grafica los 3 planos definidos por Ax=b y marca la solución."""
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # Rango de visualización centrado en la solución
    spread = max(4.0, float(np.max(np.abs(sol))) * 0.8)
    x_range = np.linspace(sol[0] - spread, sol[0] + spread, 35)
    y_range = np.linspace(sol[1] - spread, sol[1] + spread, 35)
    X, Y = np.meshgrid(x_range, y_range)

    colors = ["#4e9af1", "#e07b39", "#6ef14e"]
    labels = ["Ecuación 1", "Ecuación 2", "Ecuación 3"]

    for i in range(3):
        a0, a1, a2 = A[i]
        try:
            if abs(a2) > 1e-10:
                Z = (b[i] - a0 * X - a1 * Y) / a2
                ax.plot_surface(X, Y, Z, alpha=0.30, color=colors[i], label=labels[i])
            elif abs(a1) > 1e-10:
                z_range = np.linspace(sol[2] - spread, sol[2] + spread, 35)
                Xp, Zp = np.meshgrid(x_range, z_range)
                Yp = (b[i] - a0 * Xp) / a1
                ax.plot_surface(Xp, Yp, Zp, alpha=0.30, color=colors[i], label=labels[i])
        except Exception:
            pass

    ax.scatter(*sol, color="red", s=100, zorder=10,
               label=f"Solución ({sol[0]:.3f}, {sol[1]:.3f}, {sol[2]:.3f})")
    ax.set_xlabel("x₁ (Tránsito)")
    ax.set_ylabel("x₂ (Vuelo)")
    ax.set_zlabel("x₃ (Cobertura)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def plot_convergence(results_dict, tol=1e-6):
    """Gráfica Error vs Iteración para todos los métodos."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, data in results_dict.items():
        errors = data["errors"]
        if not errors:
            continue
        color = METHOD_COLORS.get(name, "#aaaaaa")
        ax.semilogy(range(1, len(errors) + 1), errors,
                    label=f"{name} ({len(errors)} iter.)",
                    color=color, linewidth=2)

    ax.axhline(y=tol, color="gray", linestyle="--", linewidth=1.2, label=f"Tolerancia {tol:.0e}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error ‖eₖ‖∞  (escala log)")
    ax.set_title("Convergencia comparativa de métodos iterativos")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
