# hbar = 1, m = 1
# TISE: -psi'' + V(x)*psi = E*psi
 
import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-15.0, 15.0, 4000)
h = x[1] - x[0]
N = len(x)
 
V0 = 8.0   # well depth
a  = 1.5   # well half-width
 
Vx   = np.where(np.abs(x) < a, -V0, 0.0)
diag = 1.0/h**2 + Vx
offd = -0.5/h**2 * np.ones(N - 1)
H    = np.diag(diag) + np.diag(offd, 1) + np.diag(offd, -1)
 
evals, evecs = np.linalg.eigh(H)
 
bound_idx = np.where(evals < 0)[0]
bound_E   = evals[bound_idx]
 
print(f"Finite square well  V0={V0}, a={a}")
print(f"Bound states: {len(bound_E)}")
for n, E in enumerate(bound_E):
    print(f"  n={n}:  E = {E:.5f}")
 
fig, ax = plt.subplots(figsize=(9, 6))
 
ax.plot(x, Vx, 'k-', lw=2, label='V(x)')
ax.fill_between(x, Vx, 0, color='gray', alpha=0.10)
 
for n, idx in enumerate(bound_idx):
    E   = evals[idx]
    psi = evecs[:, idx]
    psi = psi / np.sqrt(np.trapezoid(psi**2, x))
    if psi[N // 2] < 0:
        psi *= -1
 
    ax.axhline(E, linestyle='--', lw=1.2, color=f'C{n}', label=f'E{n} = {E:.3f}')
    ax.plot(x, psi + E, lw=1.8, color=f'C{n}', label=f'ψ{n}')
 
ax.set_xlim(-4, 4)
ax.set_ylim(-V0 - 1, 2)
ax.set_xlabel('x')
ax.set_ylabel('Energy')
ax.set_title(f'Finite Square Well  (V0={V0}, a={a})')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('finite_well.png', dpi=150, bbox_inches='tight')

plt.show()
