"""Numerical checks of the exact identities used in ../concomitant_weyl_commutator.md"""
import numpy as np
rng = np.random.default_rng(0)

# (1) pair phase of a quadratic field is bilinear in (rbar, Delta):  Phi(r)-Phi(r') = rbar^T M Delta + b.Delta
M = rng.normal(size=(3, 3)); M = (M + M.T) / 2; b = rng.normal(size=3)
Phi = lambda r: 0.5 * r @ M @ r + b @ r
r, rp = rng.normal(size=3), rng.normal(size=3)
rbar, D = (r + rp) / 2, r - rp
print("(1) |Phi(r)-Phi(r') - (rbar^T M D + b.D)| =", abs(Phi(r) - Phi(rp) - (rbar @ M @ D + b @ D)))

# (2) [X_a, T[g]] = T[Delta_a g]  (T = Toeplitz/convolution operator, X = multiplication by coordinate)
n = 64; x = np.arange(n, dtype=float)
g = lambda d: np.exp(1j * 0.7 * d) * np.exp(-(d / 9.0) ** 2)          # arbitrary kernel
T = lambda gf: gf(x[:, None] - x[None, :])
X = np.diag(x)
lhs = X @ T(g) - T(g) @ X
rhs = T(lambda d: d * g(d))
print("(2) ||[X,T[g]] - T[Delta g]|| =", np.linalg.norm(lhs - rhs))

# (3) Hermitian, nested commutators: [X,[X,T[g]]] = T[Delta^2 g]
lhs2 = X @ (lhs) - (lhs) @ X
print("(3) ||[X,[X,T]] - T[Delta^2 g]|| =", np.linalg.norm(lhs2 - T(lambda d: d * d * g(d))))

# (4) exact conjugation by a quadratic phase: Q T[g] Q^H = Weyl symbol sheared.  Check kernel:
#     (Q T Q^H)(r,r') = g(r-r') exp(-i(Phi(r)-Phi(r'))) = g(D) exp(-i(rbar M D + b D)) ;  first-order term = -i [Phi(X), T[g]]
M1 = 2e-4; phi = lambda r: 0.5 * M1 * r ** 2
Q = np.diag(np.exp(-1j * phi(x)))
exact = Q @ T(g) @ Q.conj().T
first = T(g) - 1j * (np.diag(phi(x)) @ T(g) - T(g) @ np.diag(phi(x)))
print("(4) ||Q T Q^H - (T - i[Phi,T])|| / ||Q T Q^H - T|| =", np.linalg.norm(exact - first) / np.linalg.norm(exact - T(g)))
# second-order term uses the symmetric nested commutator: -1/2 [Phi,[Phi,T]]
H = np.diag(phi(x)); c1 = H @ T(g) - T(g) @ H; c2 = H @ c1 - c1 @ H
second = first - 0.5 * c2
print("    with 2nd-order term:", np.linalg.norm(exact - second) / np.linalg.norm(exact - T(g)))
