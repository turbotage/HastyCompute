import numpy as np
import itertools
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math


# =========================================================
# INPUT MATRIX
# =========================================================
A = np.array([
    [-1, -1, -1],
    [ 1,  1, -1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [-0.7, -0.7, -0.7]   # try removing/adding this row
], dtype=float)

venc = 70.0
A = (np.pi / (math.sqrt(3) * venc)) * A


# =========================================================
# STEP 1: build ambiguity lattice basis via integer sampling
# (robust numeric substitute for full HNF in practice)
# =========================================================
A_pinv = np.linalg.pinv(A)

m = A.shape[0]

def lattice_vectors(K=3):
    vecs = []
    ks = list(itertools.product(range(-K, K+1), repeat=m))

    for k in ks:
        k = np.array(k)
        if np.all(k == 0):
            continue

        x = A_pinv @ k
        vecs.append(x)

    return np.array(vecs)


# =========================================================
# STEP 2: extract stable independent generators (Gram-Schmidt)
# =========================================================
def orthonormal_basis(vectors, tol=1e-8):
    basis = []

    for v in vectors:
        v = v.copy()

        for b in basis:
            v = v - np.dot(v, b) * b

        n = np.linalg.norm(v)
        if n > tol:
            basis.append(v / n)

        if len(basis) == 3:
            break

    return basis


candidates = lattice_vectors(K=3)

# sort by norm (important for stability)
candidates = sorted(candidates, key=np.linalg.norm)

basis = orthonormal_basis(candidates)

if len(basis) < 3:
    print("Warning: lattice is effectively lower-dimensional.")
    print("Basis vectors found:", len(basis))


# =========================================================
# STEP 3: generate lattice points in x-space
# =========================================================
B = np.column_stack(basis)  # 3xr (r ≤ 3)

K2 = 2
pts = []

for k in itertools.product(range(-K2, K2+1), repeat=B.shape[1]):
    k = np.array(k)
    pts.append(B @ k)

pts = np.array(pts)

# symmetry
pts = np.vstack([pts, -pts])


# =========================================================
# STEP 4: remove degeneracy (VERY IMPORTANT for Qhull)
# =========================================================
pts = pts + 1e-10 * np.random.randn(*pts.shape)


# =========================================================
# STEP 5: convex hull (Voronoi approximation dual)
# =========================================================
hull = ConvexHull(pts)
verts = pts[hull.vertices]

print("Vertices:", len(verts))


# =========================================================
# STEP 6: plotting with edges (robust)
# =========================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(verts[:,0], verts[:,1], verts[:,2], c='red')

# faces
for simplex in hull.simplices:
    tri = verts[simplex]
    ax.add_collection3d(Poly3DCollection([tri], alpha=0.2))

# edges
edges = set()
for simplex in hull.simplices:
    for i in range(3):
        a = simplex[i]
        b = simplex[(i+1) % 3]
        edges.add(tuple(sorted((a, b))))

for a, b in edges:
    pa, pb = verts[a], verts[b]
    ax.plot(
        [pa[0], pb[0]],
        [pa[1], pb[1]],
        [pa[2], pb[2]],
        color='black',
        linewidth=1
    )

ax.set_title("Lattice Quotient Injectivity Polytope (Stable Approx)")
ax.set_box_aspect([1,1,1])

plt.show()