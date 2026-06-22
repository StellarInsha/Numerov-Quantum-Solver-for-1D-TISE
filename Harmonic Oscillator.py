import numpy as np
import matplotlib.pyplot as plt



x_min, x_max = -8.0, 8.0
N = 1000

x = np.linspace(x_min, x_max, N)   
h = x[1] - x[0]                     # doing some grid spacing here



def V(x):
    return 0.5 * x**2               # harmonic oscillator t: E_n = (n + 0.5)x**2
    # return 0.5*x**2 + 0.1*x**4   # anharmonic (we have no exact answer — computer only)
    # return (x**2 - 2.0)**2        # double well (quantum tunneling happens here)
    


# Building the Hamiltonian matrix H instead of solving the differential operator H
# In quantum mechanics: H = T + V
# T = kinetic energy = -(ħ²/2m) d²/dx² (in natural units ħ=1, m=1/2 → T = -d²/dx²)\
# V here is trivial

# The second derivative d²ψ/dx² ≈ (ψ[i-1] - 2ψ[i] + ψ[i+1]) / h² by Taylor Expansion
# So -½ * d²ψ/dx² ≈ ψ[i]/h² - ψ[i-1]/(2h²) - ψ[i+1]/(2h²)

# Written as a matrix, H is tridiagonal cause the second derivative at grid point i only involves the neighboring points i-1 and i+1
#   diagonal entry [i,i]   = 1/h² + V(x[i])
#   off-diagonal   [i,i±1] = -1/(2h²)



main_diagonal = 1.0/h**2 + V(x)              # length N
off_diagonal  = -0.5/h**2 * np.ones(N - 1)   # length N-1

H = (np.diag(main_diagonal)
   + np.diag(off_diagonal, k=+1)
   + np.diag(off_diagonal, k=-1))


# Solving H ψ = E ψ
# np.linalg.eigh solves the eigenvalue problem for a matrix.
# Returns eigenvalues (energies) in ascending order,
# and eigenvectors (wavefunctions) as columns of the matrix.

eigenvalues, eigenvectors = np.linalg.eigh(H)

# eigenvectors[:, n] is the nth wavefunction as a length-N array
# Normalize each one so ∫|ψ|² dx = 1
wavefunctions = []
for n in range(8):
    psi  = eigenvectors[:, n]
    norm = np.sqrt(np.trapezoid(psi**2, x))
    wavefunctions.append(psi / norm)




# ── Printing the verification table ──────────────────────────────────────────────────
print(f"{'n':>3}  {'Numerical E':>13}  {'Exact E':>9}  {'Error':>10}")
print("─" * 42)
for n in range(8):
    E_num   = eigenvalues[n]
    E_exact = n + 0.5
    print(f"{n:>3}  {E_num:>13.8f}  {E_exact:>9.4f}  {abs(E_num - E_exact):>10.2e}")





# Plotting
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Quantum Harmonic Oscillator — Matrix Diagonalization", fontsize=13)

# Left panel: potential curve + horizontal energy level lines
ax = axes[0]
ax.plot(x, V(x), color='black', lw=2.5, label='V(x) = ½x²')
ax.fill_between(x, V(x), alpha=0.06, color='gray')
for n in range(5):
    E = eigenvalues[n]
    ax.axhline(y=E, color=f'C{n}', linestyle='--', lw=1.8, alpha=0.85,
               label=f'E_{n} = {E:.4f}')
ax.set_xlim(-6, 6);  ax.set_ylim(-0.3, 7)
ax.set_xlabel('x');  ax.set_ylabel('Energy (ℏω)')
ax.set_title('Potential and energy levels')
ax.legend(fontsize=8.5);  ax.grid(alpha=0.25)

# Right panel: wavefunctions, each lifted up to sit at its own energy level
ax = axes[1]
ax.plot(x, V(x), color='black', lw=1.5, alpha=0.35, label='V(x)')
ax.fill_between(x, V(x), alpha=0.06, color='gray')
for n in range(5):
    E   = eigenvalues[n]
    psi = wavefunctions[n]
    ax.plot(x, psi * 0.55 + E, color=f'C{n}', lw=1.8, label=f'ψ_{n}')
    ax.axhline(y=E, color=f'C{n}', linestyle=':', lw=0.8, alpha=0.5)
ax.set_xlim(-6, 6);  ax.set_ylim(-0.3, 7)
ax.set_xlabel('x')
ax.set_title('Wavefunctions (each raised to its energy level)')
ax.legend(fontsize=9);  ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('numerov_result.png', dpi=150, bbox_inches='tight')
plt.show()


