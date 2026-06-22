import numpy as np
import matplotlib.pyplot as plt

#Units h bar=1, m=1 let's say
#TISE becomes: -psi'' + V(x)*psi = E*psi

xmin,xmax,space=-15,15,4000
x=np.linspace(xmin,xmax,space)
N=len(x)

h=x[1]-x[0]

# V(x) = -alpha * delta(x)

alpha=2.0

#building the stand matrix to solve the potential

diag = 1.0/h**2 * np.ones(N)
offd = -0.5/h**2 * np.ones(N - 1)

H = np.diag(diag) + np.diag(offd, 1) + np.diag(offd, -1)

# The delta function cannot sit on a grid as-is (infinite height, zero width)
# To fix that: its entire strength alpha gets folded into the single grid point at x=0
# That one diagonal entry gets -alpha/h added to it

i0 = np.argmin(np.abs(x))
H[i0, i0] += -alpha / h

evals_d,evecs_d= np.linalg.eigh(H)
E0_num   = evals_d[0]
E0_exact = -alpha**2 / 2
 
psi0 = evecs_d[:, 0]
psi0 = psi0 / np.sqrt(np.trapezoid(psi0**2, x))
 
print("Delta potential")
print(f"  Numerical : {E0_num:.6f}")
print(f"  Exact     : {E0_exact:.6f}")
print(f"  Error     : {abs(E0_num - E0_exact):.2e}")
print(f"  Bound states: {np.sum(evals_d < 0)}  (Griffiths says exactly 1)")

#Plotting the graph

fig,ax =plt.subplots(figsize=(7,5))


ax.plot(x, psi0, color='C0', lw=2, label=f'ψ₀,  E = {E0_num:.4f}')
ax.axvline(0, color='gray', linestyle=':')
ax.set_xlim(-6, 6)
ax.set_title(f'Delta potential\nExact E = -α²/2 = {E0_exact:.4f}', fontsize=10)
ax.set_xlabel('x');  ax.legend();  ax.grid(alpha=0.25)

plt.show()
