import numpy as np
import matplotlib.pyplot as plt

#Units h bar=1, m=1 let's say
#TISE becomes: -psi'' + V(x)*psi = E*psi

xmin,xmax,space=-15,15,4000
x=np.linspace(xmin,xmax,space)
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

evalsd,evecsd=np.linalg.eigh(H)
