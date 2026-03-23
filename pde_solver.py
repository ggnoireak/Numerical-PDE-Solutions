import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})

CORES = ["#1e3a5f", "#0d6efd", "#00d4ff", "#ffffff", "#ff6b35", "#ff0055"]
CMAP_CALOR = LinearSegmentedColormap.from_list("calor", CORES)
CMAP_ONDA  = LinearSegmentedColormap.from_list("onda",  ["#0d1117", "#7c3aed", "#a78bfa", "#e0d7ff"])
CMAP_LAPLACE = "inferno"

def resolver_calor(nx=200, nt=500, alpha=0.01, snapshots=5):
    dx = 1.0 / (nx - 1)
    dt = 0.4 * dx**2 / alpha
    r  = alpha * dt / dx**2

    print(f"[Heat] r = {r:.4f} (must be ≤ 0.5 to be stable)")

    x = np.linspace(0, 1, nx)

    u = np.exp(-200 * (x - 0.5)**2)

    u[0] = u[-1] = 0.0

    passos_snap = [int(i * nt / (snapshots - 1)) for i in range(snapshots)]
    historico   = {}

    for n in range(nt):
        if n in passos_snap:
            historico[n] = (u.copy(), n * dt)

        u[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u[0] = u[-1] = 0.0

    if nt - 1 not in historico:
        historico[nt - 1] = (u.copy(), (nt - 1) * dt)

    return x, historico, r


def resolver_onda(nx=300, nt=600, c=1.0, snapshots=5):
    dx  = 1.0 / (nx - 1)
    dt  = 0.9 * dx / c
    cfl = c * dt / dx

    print(f"[Wave] CFL = {cfl:.4f} (must be ≤ 1 to be stable)")

    x = np.linspace(0, 1, nx)

    u_prev = np.exp(-500 * (x - 0.3)**2)
    u_curr = u_prev.copy()
    u_next = np.zeros(nx)

    u_prev[0] = u_prev[-1] = 0.0
    u_curr[0] = u_curr[-1] = 0.0

    passos_snap = [int(i * nt / (snapshots - 1)) for i in range(snapshots)]
    historico   = {0: (u_curr.copy(), 0.0)}

    for n in range(1, nt):
        u_next[1:-1] = (2*u_curr[1:-1] - u_prev[1:-1]
                        + cfl**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
        u_next[0] = u_next[-1] = 0.0

        if n in passos_snap:
            historico[n] = (u_next.copy(), n * dt)

        u_prev, u_curr = u_curr, u_next.copy()

    return x, historico, cfl


def resolver_laplace(n=80, max_iter=5000, tol=1e-5, snapshots=4):
    u = np.zeros((n, n))

    u[0,  :]  = 100.0
    u[-1, :]  = 0.0
    u[:,  0]  = 0.0
    u[:, -1]  = 0.0

    iters_snap = sorted(set([1, 10, 50, 200, max_iter]))[:snapshots]
    historico  = {}
    residuos   = []

    for k in range(1, max_iter + 1):
        u_old = u.copy()

        u[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1]
                                + u[1:-1, :-2] + u[1:-1, 2:])

        u[0, :]  = 100.0
        u[-1, :] = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0

        res = np.max(np.abs(u - u_old))
        residuos.append(res)

        if k in iters_snap:
            historico[k] = u.copy()

        if res < tol:
            print(f"[Laplace] Converged in {k} iterations (residual = {res:.2e})")
            historico[k] = u.copy()
            break
    else:
        print(f"[Laplace] Max iterations reached (final residual = {residuos[-1]:.2e})")

    return u, historico, residuos


def plotar_calor(x, historico, r):
    passos = sorted(historico.keys())
    n = len(passos)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    fig.suptitle(f"Heat Equation — ∂u/∂t = α·∂²u/∂x²   (r = {r:.3f})",
                 fontsize=13, color="#58a6ff", y=1.02)

    for ax, p in zip(axes, passos):
        u, t = historico[p]
        ax.fill_between(x, u, alpha=0.3, color="#0d6efd")
        ax.plot(x, u, color="#00d4ff", lw=2)
        ax.set_title(f"t = {t:.4f} s")
        ax.set_xlabel("x")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)

    axes[0].set_ylabel("u(x, t) [temperature]")
    fig.tight_layout()
    return fig


def plotar_onda(x, historico, cfl):
    passos = sorted(historico.keys())
    n = len(passos)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    fig.suptitle(f"Wave Equation — ∂²u/∂t² = c²·∂²u/∂x²   (CFL = {cfl:.3f})",
                 fontsize=13, color="#a78bfa", y=1.02)

    for ax, p in zip(axes, passos):
        u, t = historico[p]
        ax.fill_between(x, u, alpha=0.25, color="#7c3aed")
        ax.plot(x, u, color="#c4b5fd", lw=2)
        ax.set_title(f"t = {t:.4f} s")
        ax.set_xlabel("x")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)

    axes[0].set_ylabel("u(x, t) [displacement]")
    fig.tight_layout()
    return fig


def plotar_laplace(historico, residuos):
    iters = sorted(historico.keys())
    n_snap = len(iters)

    fig = plt.figure(figsize=(5*n_snap + 5, 5))
    gs  = gridspec.GridSpec(1, n_snap + 1, figure=fig, wspace=0.4)
    fig.suptitle("Laplace Equation — ∂²u/∂x² + ∂²u/∂y² = 0 (Gauss-Seidel)",
                 fontsize=13, color="#ffa657", y=1.02)

    for i, it in enumerate(iters):
        ax  = fig.add_subplot(gs[0, i])
        im  = ax.imshow(historico[it], cmap=CMAP_LAPLACE,
                        origin="lower", vmin=0, vmax=100)
        ax.set_title(f"iter = {it}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, label="T")

    ax_res = fig.add_subplot(gs[0, -1])
    ax_res.semilogy(residuos, color="#ffa657", lw=1.5)
    ax_res.set_title("Convergence")
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Max residual")
    ax_res.grid(True)

    fig.tight_layout()
    return fig


if __name__ == "__main__":

    print("=" * 55)
    print("  Numerical Solution of PDEs — Finite Differences")
    print("=" * 55)
    print("\nSelect which equation to solve:")
    print("  1. Heat Equation (diffusion)")
    print("  2. Wave Equation (propagation)")
    print("  3. Laplace Equation (steady-state)")
    print("  4. Solve all equations")
    print("  0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == "1":
            print("\n→ Solving Heat Equation...")
            x_c, hist_c, r = resolver_calor(nx=200, nt=800, alpha=0.01, snapshots=5)
            fig_c = plotar_calor(x_c, hist_c, r)
            fig_c.savefig("calor.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: calor.png")
            plt.show()
            
        elif choice == "2":
            print("\n→ Solving Wave Equation...")
            x_o, hist_o, cfl = resolver_onda(nx=300, nt=600, c=1.0, snapshots=5)
            fig_o = plotar_onda(x_o, hist_o, cfl)
            fig_o.savefig("onda.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: onda.png")
            plt.show()
            
        elif choice == "3":
            print("\n→ Solving Laplace Equation...")
            u_l, hist_l, res_l = resolver_laplace(n=80, max_iter=5000, tol=1e-5, snapshots=4)
            fig_l = plotar_laplace(hist_l, res_l)
            fig_l.savefig("laplace.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: laplace.png")
            plt.show()
            
        elif choice == "4":
            print("\n→ Solving Heat Equation...")
            x_c, hist_c, r = resolver_calor(nx=200, nt=800, alpha=0.01, snapshots=5)
            fig_c = plotar_calor(x_c, hist_c, r)
            fig_c.savefig("calor.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: calor.png")
            
            print("\n→ Solving Wave Equation...")
            x_o, hist_o, cfl = resolver_onda(nx=300, nt=600, c=1.0, snapshots=5)
            fig_o = plotar_onda(x_o, hist_o, cfl)
            fig_o.savefig("onda.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: onda.png")
            
            print("\n→ Solving Laplace Equation...")
            u_l, hist_l, res_l = resolver_laplace(n=80, max_iter=5000, tol=1e-5, snapshots=4)
            fig_l = plotar_laplace(hist_l, res_l)
            fig_l.savefig("laplace.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
            print("  Saved: laplace.png")
            
            print("\n✓ All graphs generated successfully!")
            plt.show()
            
        elif choice == "0":
            print("Exiting...")
            
        else:
            print("Invalid option. Please run again.")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
