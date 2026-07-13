# Hamiltonian Matrix Schrödinger Solver

Most potentials in a quantum mechanics course are chosen because they happen to be analytically solvable. The harmonic oscillator, the finite well, the delta potential Griffiths can solve these by hand because they have special mathematical structure. But if we change the potential slightly, the entire analytical method breaks down into pieces. 

This Project is an attempt to solve such potentials.

I have done it by turning the Schrödinger equation into a matrix problem. Diagonaliinge the matrix where the eigenvalues are your energy levels and The eigenvectors are your wavefunctions. It works for any potential — solvable or not.

---

## The Idea

The time-independent Schrödinger equation is:

```
−ψ''(x) + V(x)ψ(x) = Eψ(x)
```

This is an eigenvalue equation. H acting on ψ gives back ψ scaled by E. The question is: how do you represent H as a matrix?

The key is the second derivative. On a discrete grid of N points spaced h apart, the second derivative at point i is approximately:

```
ψ''(xᵢ) ≈ (ψ[i−1] − 2ψ[i] + ψ[i+1]) / h²
```

This is just Taylor expansion — expand ψ(x+h) and ψ(x−h), add them, and then rearrange.

Written for every grid point simultaneously, this becomes a matrix equation. The Hamiltonian H = T + V is a tridiagonal matrix:

```
diagonal entry [i,i]   =  1/h²  +  V(xᵢ)       ← kinetic + potential
off-diagonal   [i,i±1] = −1/(2h²)               ← coupling to neighbours
```

Handing this matrix to `numpy`. It returns all eigenvalues (energies) and eigenvectors (wavefunctions) in one call. The whole Schrödinger equation gets reduced to thirty lines of code.

---

## Why This Works

Think of the wavefunction as a vector — N numbers, one per grid point. The Hamiltonian tells you how each point is connected to its neighbours through the kinetic energy, and how each point is shifted by the local potential. Solving Hψ = Eψ means finding the special vectors that H stretches without rotating. Those are the eigenvectors. Their stretch factors are the eigenvalues (the allowed energies)

This is identical in spirit to what Griffiths does analytically with ladder operators for the harmonic oscillator. The difference is that the analytical method exploits a specific algebraic property of V = ½x² that most potentials do not have. The matrix method exploits nothing special about V at all. It works universally.

---

## Usage

```bash
pip install numpy matplotlib scipy
python qm_solver.py
```

Enter any potential V(x) using `x` as the variable. Available functions: `sin`, `cos`, `exp`, `abs`, `sqrt`, `log`, `pi`, `np`.

---

## Potentials

| # | Potential | Expression |
|---|-----------|------------|
| 1 | Harmonic oscillator | `0.5*x**2` |
| 2 | Finite square well | `np.where(np.abs(x)<2, -8.0, 0.0)` |
| 3 | Double well | `(x**2 - 2)**2` |
| 4 | Anharmonic oscillator | `0.5*x**2 + 0.1*x**4` |
| 5 | Linear potential | `np.abs(x)` |
| 6 | Soft Coulomb | `-(1/(np.abs(x)+0.1))` |
| c | Custom | anything |

---

## Output

For any potential you enter, the solver produces:

1. **Energy levels** — the potential curve with each allowed energy marked as a horizontal line, exactly as drawn in textbook diagrams
2. **Wavefunctions** — each ψₙ placed at its energy level; the number of nodes increases by one with each state, a consequence of the oscillation theorem
3. **Probability densities** — |ψₙ|² showing where the particle is most likely to be found; notice how the ground state avoids the classical turning points while higher states approach the classical distribution
4. **Energy spectrum** — bar chart of eigenvalues; equal spacing signals the harmonic oscillator, unequal spacing signals anharmonicity
5. **Time evolution** — animated probability density for the superposition state ψ(t) = (ψ₀e⁻ⁱᴱ⁰ᵗ + ψ₁e⁻ⁱᴱ¹ᵗ)/√2, oscillating with period T = 2π/(E₁−E₀)

---

## Verification

For V(x) = ½x², Griffiths derives Eₙ = n + ½ analytically. Numerical results with N = 1000 grid points:

| n | Numerical | Exact (n + ½) | Error |
|---|-----------|---------------|-------|
| 0 | 0.499992 | 0.5000 | 8.0e-06 |
| 1 | 1.499960 | 1.5000 | 4.0e-05 |
| 2 | 2.499896 | 2.5000 | 1.0e-04 |
| 3 | 3.499800 | 3.5000 | 2.0e-04 |
| 4 | 4.499671 | 4.5000 | 3.3e-04 |

Errors grow with n because higher states oscillate more rapidly — more nodes means the wavefunction varies faster, which requires a finer grid to represent accurately. Error scales as h², consistent with the second-order finite difference approximation.

---

## Interesting Cases

**Double well** `V = (x²−2)²`

Two symmetric wells separated by a central barrier. Classically, a particle in the left well stays there. Quantum mechanically, it does not. The two lowest eigenstates are nearly degenerate — a symmetric combination ψ₊ and an antisymmetric combination ψ₋. The energy splitting ΔE between them is nonzero precisely because the particle has amplitude to tunnel through the barrier. If the barrier were infinitely high, the two states would be exactly degenerate and tunneling would vanish. The splitting is a direct measure of the tunneling amplitude.

This same physics governs the ammonia molecule, where the nitrogen atom tunnels between two positions, and superconducting qubits, where the two-well structure of the Josephson potential defines the qubit states.

**Anharmonic oscillator** `V = ½x² + 0.1x⁴`

No closed-form solution exists for any nonzero coefficient of x⁴. The x⁴ term makes the potential walls steeper at large x. Higher energy states, which extend further from the origin, feel this steeper wall more strongly so their energies are pushed up more than lower states. The equal spacing of the harmonic oscillator breaks down. This unequal spacing is physically important: real molecular potentials are never perfect parabolas, and the anharmonic correction explains why molecular vibrational spectra are not perfectly evenly spaced.

**Delta potential** `V = −αδ(x)`

Approximated numerically by concentrating the potential at a single grid point: `H[i₀, i₀] += −α/h`. As h → 0 this converges to the true delta function result. Griffiths shows analytically that there is exactly one bound state with energy E = −α²/2. The numerical result reproduces this to within 1e-4.

---


---

## Requirements

```
numpy
matplotlib
scipy
```

## Personal Remarks
---
This project was an attempt to implement my understanding of Quantum Mechanics into something worthwhile. 
Alongside self study of Griffiths, producing the exact same answers made ne understand the subject to it's core.
