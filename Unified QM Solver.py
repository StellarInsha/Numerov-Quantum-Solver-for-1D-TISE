"""
1D Quantum Mechanics Solver

Enter any potential V(x). The solver:
  - Finds all bound state energies (Hamiltonian matrix diagonalization)
  - Plots wavefunctions and probability densities
  - Shows energy spectrum as a bar chart
  - Animates the time evolution of a quantum superposition state

Units: hbar = 1,  m = 1 (for ease)
Method: Finite difference Hamiltonian matrix and numpy eigenvalue solver
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh_tridiagonal
import sys


# ── PRESETS ───────────────────────────────────────────────────────────────────

PRESETS = {
    "1": ("0.5*x**2",
          "Harmonic Oscillator  ✓ Griffiths verified",
          -8, 8),
    "2": ("np.where(np.abs(x)<2, -8.0, 0.0)",
          "Finite Square Well",
          -8, 8),
    "3": ("(x**2 - 2)**2",
          "Double Well  [quantum tunneling]",
          -5, 5),
    "4": ("0.5*x**2 + 0.1*x**4",
          "Anharmonic Oscillator  [unsolvable by hand]",
          -8, 8),
    "5": ("np.abs(x)",
          "Linear Potential  [like gravity]",
          -8, 8),
    "6": ("-(1/(np.abs(x)+0.1))",
          "Soft Coulomb  [hydrogen-like]",
          -10, 10),
}

COLORS = ['#00d4ff','#ff6b6b','#51cf66','#ffd43b',
          '#cc5de8','#ff922b','#20c997','#f06595']

BG     = '#0f0f1a'
GRID   = '#222233'


# ── CORE SOLVER ───────────────────────────────────────────────────────────────

def solve(V_string, x_min, x_max, N=1000):
    """
    Discretize H = -d²/dx² + V(x) as a tridiagonal matrix.
    Find eigenvalues (energies) and eigenvectors (wavefunctions).
    """
    x = np.linspace(x_min, x_max, N)
    h = x[1] - x[0]

    safe = {
        "x": x, "np": np,
        "sin": np.sin,  "cos": np.cos,  "exp": np.exp,
        "abs": np.abs,  "sqrt": np.sqrt, "pi": np.pi,
        "log": np.log,  "tan": np.tan,
        "__builtins__": {}
    }

    try:
        Vx = eval(V_string, {"__builtins__": {}}, safe)
        Vx = np.asarray(Vx, dtype=float)
        if Vx.shape == ():
            Vx = Vx * np.ones(N)
    except Exception as e:
        print(f"\nCould not evaluate V(x) = '{V_string}'\nError: {e}")
        sys.exit(1)

    # H = T + V
    # T diagonal: 1/h²,  T off-diagonal: -1/(2h²)
    diag = 1.0/h**2 + Vx
    offd = -0.5/h**2 * np.ones(N - 1)

    evals, evecs = eigh_tridiagonal(diag, offd)

    # Normalize and fix sign of each wavefunction
    psis = []
    for n in range(min(8, N)):
        psi  = evecs[:, n]
        norm = np.sqrt(np.trapezoid(psi**2, x))
        psi  = psi / norm
        if psi[N//2] < 0:
            psi *= -1
        psis.append(psi)

    return x, Vx, evals[:8], psis


# ── GRIFFITHS VERIFICATION ────────────────────────────────────────────────────

def griffiths_check(evals, V_string):
    if "0.5*x**2" in V_string and "x**4" not in V_string:
        print("\n  Griffiths verification — Harmonic oscillator (exact: Eₙ = n + ½)")
        print(f"  {'n':>3}  {'Numerical':>12}  {'Exact':>8}  {'Error':>10}")
        print("  " + "─"*38)
        for n in range(min(6, len(evals))):
            exact = n + 0.5
            print(f"  {n:>3}  {evals[n]:>12.6f}  {exact:>8.4f}  {abs(evals[n]-exact):>10.2e}")


# ── PLOT + ANIMATION ──────────────────────────────────────────────────────────

def run(V_string, x_min, x_max, n_show=5):

    print(f"\nSolving V(x) = {V_string} ...")
    x, Vx, evals, psis = solve(V_string, x_min, x_max)

    # Print energies
    print(f"\n  {'n':>3}  {'Energy':>12}")
    print("  " + "─"*18)
    for n in range(min(n_show, len(evals))):
        tag = "  ← bound" if evals[n] < 0 else ""
        print(f"  {n:>3}  {evals[n]:>12.6f}{tag}")

    griffiths_check(evals, V_string)

    # ── Figure layout ──
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Quantum Mechanics Solver  —  V(x) = {V_string}",
                 color='white', fontsize=12, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)   # potential + levels
    ax2 = fig.add_subplot(2, 3, 2)   # wavefunctions
    ax3 = fig.add_subplot(2, 3, 3)   # probability densities
    ax4 = fig.add_subplot(2, 3, 4)   # energy bar chart
    ax5 = fig.add_subplot(2, 3, (5,6))  # animation

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor(BG)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(alpha=0.15, color=GRID)

    n_plot = min(n_show, len(evals))
    xlim   = (x_min*0.65, x_max*0.65)
    E_lo   = min(np.min(Vx), evals[0]) - 0.5
    E_hi   = evals[n_plot-1] + 1.5
    ylim   = (E_lo, E_hi)

    # ── Panel 1: Potential + energy levels ──
    ax1.plot(x, Vx, color='white', lw=2, label='V(x)')
    ax1.fill_between(x, Vx, E_lo, alpha=0.12, color='white')
    for n in range(n_plot):
        if E_lo < evals[n] < E_hi:
            ax1.axhline(evals[n], color=COLORS[n], ls='--', lw=1.5,
                        label=f'E{n}={evals[n]:.2f}')
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
    ax1.set_title('Potential  &  Energy Levels', color='white', fontsize=9)
    ax1.set_xlabel('x', color='white'); ax1.set_ylabel('Energy', color='white')
    ax1.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 2: Wavefunctions ──
    ax2.plot(x, Vx, color='white', lw=1, alpha=0.25)
    for n in range(n_plot):
        if E_lo < evals[n] < E_hi:
            ax2.plot(x, psis[n]*0.45 + evals[n], color=COLORS[n],
                     lw=1.8, label=f'ψ{n}')
            ax2.axhline(evals[n], color=COLORS[n], ls=':', lw=0.7, alpha=0.35)
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim)
    ax2.set_title('Wavefunctions  ψₙ(x)', color='white', fontsize=9)
    ax2.set_xlabel('x', color='white')
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 3: Probability densities ──
    for n in range(n_plot):
        prob = psis[n]**2
        ax3.plot(x, prob / np.max(prob), color=COLORS[n],
                 lw=1.8, label=f'|ψ{n}|²')
        ax3.fill_between(x, prob/np.max(prob), alpha=0.08, color=COLORS[n])
    ax3.set_xlim(*xlim)
    ax3.set_title('Probability Densities  |ψₙ(x)|²', color='white', fontsize=9)
    ax3.set_xlabel('x', color='white'); ax3.set_ylabel('|ψ|²  (normalized)', color='white')
    ax3.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 4: Energy spectrum bar chart ──
    bars = ax4.bar(range(n_plot), evals[:n_plot],
                   color=COLORS[:n_plot], edgecolor='white',
                   linewidth=0.6, alpha=0.85)
    ax4.set_xticks(range(n_plot))
    ax4.set_xticklabels([f'E{n}' for n in range(n_plot)], color='white')
    ax4.set_ylabel('Energy', color='white')
    ax4.set_title('Energy Spectrum', color='white', fontsize=9)
    for bar, E in zip(bars, evals[:n_plot]):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 E + abs(E)*0.02 + 0.05,
                 f'{E:.3f}', ha='center', va='bottom',
                 color='white', fontsize=7)

    # ── Panel 5: Animation — time evolution of superposition ──
    # Equal superposition of ground state and first excited state
    # |psi(t)|^2 oscillates between the two states

    T_period = 2*np.pi / abs(evals[1] - evals[0])
    t_vals   = np.linspace(0, 2*T_period, 80)

    def get_prob(t):
        psi_t = (psis[0] * np.exp(-1j*evals[0]*t) +
                 psis[1] * np.exp(-1j*evals[1]*t)) / np.sqrt(2)
        return np.abs(psi_t)**2

    ax5.set_xlim(*xlim)
    ax5.set_ylim(0, np.max(get_prob(0)) * 1.4)
    ax5.set_xlabel('x', color='white')
    ax5.set_ylabel('|ψ(x,t)|²', color='white')
    ax5.set_title(
        'Time Evolution  —  Superposition  ψ(t) = (ψ₀·e⁻ⁱᴱ⁰ᵗ + ψ₁·e⁻ⁱᴱ¹ᵗ)/√2',
        color='white', fontsize=9
    )

    prob0       = get_prob(0)
    line_anim,  = ax5.plot(x, prob0, color='#00d4ff', lw=2)
    fill_anim   = [ax5.fill_between(x, prob0, alpha=0.2, color='#00d4ff')]
    time_label  = ax5.text(0.02, 0.92, 't = 0.00',
                           transform=ax5.transAxes, color='white', fontsize=9)
    period_label = ax5.text(0.75, 0.92, f'T = {T_period:.2f}',
                            transform=ax5.transAxes, color='#aaa', fontsize=8)

    def animate(frame):
        t    = t_vals[frame]
        prob = get_prob(t)
        line_anim.set_ydata(prob)
        fill_anim[0].remove()
        fill_anim[0] = ax5.fill_between(x, prob, alpha=0.18, color='#00d4ff')
        time_label.set_text(f't = {t:.2f}')
        return line_anim, time_label

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    anim = animation.FuncAnimation(
        fig, animate, frames=len(t_vals), interval=50, blit=False
    )

    # Save GIF
    gif_name = 'qm_output.gif'
    anim.save(gif_name, writer='pillow', fps=20, dpi=100)
    print(f"\n  Animation saved as {gif_name}")

    # Save static preview
    png_name = 'qm_output.png'
    plt.savefig(png_name, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Static plot saved as {png_name}")

    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   1D SCHRÖDINGER EQUATION SOLVER")
    print("   Method: Hamiltonian Matrix Diagonalization")
    print("   hbar = 1,  m = 1")
    print("=" * 58)

    print("\nPreset potentials:")
    for key, (expr, desc, *_) in PRESETS.items():
        print(f"  [{key}] {desc}")
        print(f"       V(x) = {expr}")
    print("  [c] Custom potential")

    print()
    choice = input("Choose [1-6 or c]: ").strip().lower()

    if choice in PRESETS:
        V_string, desc, x_min, x_max = PRESETS[choice]
        print(f"\nSelected: {desc}")
    elif choice == 'c':
        print("\nAvailable functions: sin, cos, exp, abs, sqrt, log, pi, np")
        print("Examples:")
        print("  x**4")
        print("  sin(x)")
        print("  np.where(np.abs(x)<3, -5.0, 0.0)")
        print("  0.5*x**2 + 0.05*x**3")
        V_string = input("\nV(x) = ").strip()
        x_min    = float(input("x_min (default -10): ").strip() or -10)
        x_max    = float(input("x_max (default +10): ").strip() or 10)
    else:
        print("Defaulting to harmonic oscillator.")
        V_string, _, x_min, x_max = PRESETS["1"]

    run(V_string, x_min, x_max)


if __name__ == "__main__":
    main()
