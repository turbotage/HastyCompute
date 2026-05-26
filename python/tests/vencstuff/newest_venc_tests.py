import itertools
import math

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from sympy.matrices.normalforms import smith_normal_decomp


EXAMPLE_VENC = 35.0
EXAMPLE_A_INT = np.array(
	[
		[-2, -2, -2],
		[2, 2, -2],
		[2, -2, 2],
		[-2, 2, 2],
		[-1, -1, -1],
		[1, 1, -1],
		[1, -1, 1],
		[-1, 1, 1],
	],
	dtype=int,
)
EXAMPLE_A0_INT = np.vstack([EXAMPLE_A_INT, np.array([[0, 0, 0]], dtype=int)])
EXAMPLE_PHASE_SCALE = np.pi / (math.sqrt(3.0) * EXAMPLE_VENC)


def _validate_rank3_matrix(matrix):
	matrix = np.asarray(matrix, dtype=float)
	if matrix.ndim != 2:
		raise ValueError("A must be a 2D array.")
	if matrix.shape[1] != 3:
		raise ValueError("This helper expects a 3-column matrix for 3D velocities.")
	if np.linalg.matrix_rank(matrix) < 3:
		raise ValueError("A must have rank 3.")
	return matrix


def _validate_integer_rank3_matrix(integer_matrix):
	integer_matrix = np.asarray(integer_matrix)
	rounded = np.rint(integer_matrix).astype(int)
	if not np.allclose(integer_matrix, rounded):
		raise ValueError("The injectivity solver expects an integer-valued matrix.")
	return _validate_rank3_matrix(rounded)


def build_scaled_matrix(integer_matrix, phase_scale):
	return phase_scale * np.asarray(integer_matrix, dtype=float)


def wrap(phases):
	return (np.asarray(phases, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_encodings_match(matrix, velocity_a, velocity_b, atol=1e-8):
	matrix = np.asarray(matrix, dtype=float)
	velocity_a = np.asarray(velocity_a, dtype=float)
	velocity_b = np.asarray(velocity_b, dtype=float)
	return np.allclose(wrap(matrix @ velocity_a), wrap(matrix @ velocity_b), atol=atol)


def make_pseudoinverse_decoder(matrix):
	matrix = _validate_rank3_matrix(matrix)
	pseudoinverse = np.linalg.pinv(matrix)

	def decoder(wrapped_phase):
		return pseudoinverse @ np.asarray(wrapped_phase, dtype=float)

	return decoder


def _default_difference_decoder_rows(full_matrix, row_tol=1e-12):
	full_matrix = _validate_rank3_matrix(full_matrix)
	row_norms = np.linalg.norm(full_matrix, axis=1)
	solve_rows = [index for index, norm in enumerate(row_norms) if norm > row_tol]
	if not solve_rows:
		raise ValueError("No non-zero rows are available for velocity reconstruction.")
	if np.linalg.matrix_rank(full_matrix[solve_rows, :]) < 3:
		raise ValueError("The non-zero rows do not span a rank-3 reconstruction system.")
	return solve_rows


def make_difference_based_mri_decoder(full_matrix, solve_rows=None, estimate_rows=None, row_tol=1e-12):
	"""
	Build a general difference-based MRI decoder.

	The decoder uses one matrix to estimate expected unwrapped phases and another
	full-rank row subset to solve for velocity after rounding wrap differences:

	1. Estimate velocity from wrapped phases using `estimate_rows`.
	2. Predict the unwrapped phases on `solve_rows` from that estimate.
	3. Round the phase mismatch in units of 2π to recover integer wraps.
	4. Reconstruct velocity from the unwrapped phases on `solve_rows`.

	This generalizes the trusted 5-point decoder in `venc_tests.py`, where the
	zero row participates in phase estimation but not in the reconstruction rows.
	"""
	full_matrix = _validate_rank3_matrix(full_matrix)
	if solve_rows is None:
		solve_rows = _default_difference_decoder_rows(full_matrix, row_tol=row_tol)
	else:
		solve_rows = list(solve_rows)
	if estimate_rows is None:
		estimate_rows = list(range(full_matrix.shape[0]))
	else:
		estimate_rows = list(estimate_rows)

	solve_matrix = full_matrix[solve_rows, :]
	estimate_matrix = full_matrix[estimate_rows, :]
	if np.linalg.matrix_rank(solve_matrix) < 3:
		raise ValueError("solve_rows must define a rank-3 reconstruction matrix.")
	if np.linalg.matrix_rank(estimate_matrix) < 3:
		raise ValueError("estimate_rows must define a rank-3 estimation matrix.")

	solve_pseudoinverse = np.linalg.pinv(solve_matrix)
	estimate_pseudoinverse = np.linalg.pinv(estimate_matrix)
	estimate_to_solve = solve_matrix @ estimate_pseudoinverse

	def decoder(wrapped_phase):
		wrapped_phase = np.asarray(wrapped_phase, dtype=float)
		estimate_phase = wrapped_phase[estimate_rows]
		solve_phase = wrapped_phase[solve_rows]
		wrap_counts = np.round((estimate_to_solve @ estimate_phase - solve_phase) / (2.0 * np.pi))
		return solve_pseudoinverse @ (solve_phase + 2.0 * np.pi * wrap_counts)

	return decoder


def make_five_point_mri_decoder(full_matrix):
	"""
	Build the exact 5-point MRI decoder used in `venc_tests.py`.

	This preserves the existing decoder formula rather than replacing it with a
	minimum-norm or principal-branch surrogate.
	"""
	full_matrix = _validate_rank3_matrix(full_matrix)
	return make_difference_based_mri_decoder(
		full_matrix,
		solve_rows=list(range(full_matrix.shape[0] - 1)),
		estimate_rows=list(range(full_matrix.shape[0])),
	)


def decode_minimum_norm_from_phase(matrix, wrapped_phase, wrap_radius=4, atol=1e-7):
	"""
	Brute-force the minimum-norm decoder by enumerating wrap multiplicities.

	This is slower than the lattice construction but it is useful as a direct
	validation that higher wrap multiplicities are being handled correctly.
	"""
	matrix = _validate_rank3_matrix(matrix)
	wrapped_phase = np.asarray(wrapped_phase, dtype=float)
	pseudoinverse = np.linalg.pinv(matrix)
	best = None
	best_norm = None

	for coefficient in itertools.product(range(-wrap_radius, wrap_radius + 1), repeat=matrix.shape[0]):
		candidate = pseudoinverse @ (wrapped_phase + 2.0 * np.pi * np.asarray(coefficient, dtype=float))
		if not np.allclose(wrap(matrix @ candidate), wrapped_phase, atol=atol):
			continue

		candidate_norm = float(np.linalg.norm(candidate))
		if best is None or candidate_norm < best_norm - 1e-10:
			best = candidate
			best_norm = candidate_norm

	if best is None:
		raise ValueError("No valid candidate was found for the wrapped phase.")

	return best


def principal_branch_polytope_vertices(matrix, wrap_limit=np.pi):
	"""
	Return the principal-branch section {v : |A v| <= wrap_limit}.
	"""
	matrix = _validate_rank3_matrix(matrix)
	bounds = wrap_limit * np.ones((matrix.shape[0], 1), dtype=float)
	halfspaces = np.vstack(
		[
			np.hstack([matrix, -bounds]),
			np.hstack([-matrix, -bounds]),
		]
	)
	intersection = HalfspaceIntersection(halfspaces, np.zeros(3, dtype=float))
	return np.unique(np.round(intersection.intersections, decimals=8), axis=0)


def principal_branch_polytope_hull(matrix, wrap_limit=np.pi):
	vertices = principal_branch_polytope_vertices(matrix, wrap_limit=wrap_limit)
	return vertices, ConvexHull(vertices)


def wrap_cell_inequalities(matrix, wrap_index, wrap_limit=np.pi):
	matrix = _validate_rank3_matrix(matrix)
	wrap_index = np.asarray(wrap_index, dtype=float)
	if wrap_index.shape != (matrix.shape[0],):
		raise ValueError("wrap_index must have one entry per row of A.")
	a_ub = np.vstack([matrix, -matrix])
	b_ub = np.concatenate([wrap_limit + 2.0 * np.pi * wrap_index, wrap_limit - 2.0 * np.pi * wrap_index])
	return a_ub, b_ub


def point_is_in_wrap_cell(matrix, point, wrap_index, wrap_limit=np.pi, atol=1e-8):
	point = np.asarray(point, dtype=float)
	wrap_index = np.asarray(wrap_index, dtype=float)
	phase = np.asarray(matrix, dtype=float) @ point - 2.0 * np.pi * wrap_index
	return np.all(phase >= -wrap_limit - atol) and np.all(phase <= wrap_limit + atol)


def find_wrap_cell_interior(matrix, wrap_index, wrap_limit=np.pi, interior_margin=1e-7):
	a_ub, b_ub = wrap_cell_inequalities(matrix, wrap_index, wrap_limit=wrap_limit)
	result = linprog(np.zeros(matrix.shape[1]), A_ub=a_ub, b_ub=b_ub - interior_margin, method="highs")
	if not result.success:
		return None
	return result.x


def wrap_cell_vertices(matrix, wrap_index, wrap_limit=np.pi, interior_margin=1e-7):
	interior = find_wrap_cell_interior(matrix, wrap_index, wrap_limit=wrap_limit, interior_margin=interior_margin)
	if interior is None:
		return None
	a_ub, b_ub = wrap_cell_inequalities(matrix, wrap_index, wrap_limit=wrap_limit)
	halfspaces = np.hstack([a_ub, -b_ub[:, None]])
	intersection = HalfspaceIntersection(halfspaces, interior)
	return np.unique(np.round(intersection.intersections, decimals=8), axis=0)


def exact_recovery_cells(matrix, decoder, wrap_radius=2, wrap_limit=np.pi, offset_tol=1e-8):
	"""
	Enumerate the exact-recovery cells for a decoder D.

	For each wrap cell indexed by k with
	A v - 2π k ∈ [-π, π)^m,
	we test whether D(w(A v)) = v on that cell by evaluating the decoder at one
	strict interior point. For affine-on-cell decoders such as the pseudoinverse
	decoder and the MRI 5-point decoder from `venc_tests.py`, the decoder offset is
	constant on each cell.
	"""
	matrix = _validate_rank3_matrix(matrix)
	cells = []

	for wrap_index in itertools.product(range(-wrap_radius, wrap_radius + 1), repeat=matrix.shape[0]):
		interior = find_wrap_cell_interior(matrix, wrap_index, wrap_limit=wrap_limit)
		if interior is None:
			continue

		wrapped_phase = wrap(matrix @ interior)
		decoded = decoder(wrapped_phase)
		offset = decoded - interior
		if np.linalg.norm(offset) > offset_tol:
			continue

		vertices = wrap_cell_vertices(matrix, wrap_index, wrap_limit=wrap_limit)
		if vertices is None:
			continue
		cells.append(
			{
				"wrap_index": np.asarray(wrap_index, dtype=int),
				"interior": interior,
				"vertices": vertices,
				"hull": ConvexHull(vertices),
			}
		)

	return cells


def point_is_in_exact_recovery_cells(matrix, point, cells, wrap_limit=np.pi, atol=1e-8):
	for cell in cells:
		if point_is_in_wrap_cell(matrix, point, cell["wrap_index"], wrap_limit=wrap_limit, atol=atol):
			return True
	return False


def minimum_norm_representative(velocity, lattice_basis, coefficient_radius=4):
	"""
	Return the minimum-norm representative of the lattice coset velocity + L.

	For the injectivity rule "pick the solution with smallest ||v||", the selected
	velocity is the shortest vector in the coset modulo the ambiguity lattice.
	"""
	velocity = np.asarray(velocity, dtype=float)
	lattice_basis = np.asarray(lattice_basis, dtype=float)
	best = None
	best_norm = None

	for coefficient in itertools.product(range(-coefficient_radius, coefficient_radius + 1), repeat=3):
		shift = lattice_basis @ np.asarray(coefficient, dtype=float)
		candidate = velocity + shift
		candidate_norm = float(np.linalg.norm(candidate))
		if best is None or candidate_norm < best_norm - 1e-10:
			best = candidate
			best_norm = candidate_norm

	return best


def integer_ambiguity_basis(integer_matrix, phase_scale=1.0):
	"""
	Compute an exact basis for {Δv : wrap(phase_scale * A * (v + Δv)) = wrap(phase_scale * A * v)}.
	"""
	integer_matrix = _validate_integer_rank3_matrix(integer_matrix)
	sympy_matrix = sp.Matrix(integer_matrix.astype(int).tolist())
	diagonal, _, right_transform = smith_normal_decomp(sympy_matrix, domain=sp.ZZ)
	invariant_factors = [int(diagonal[index, index]) for index in range(3)]
	normalized_basis = right_transform * sp.diag(*[sp.Rational(1, factor) for factor in invariant_factors])
	return np.array((2 * sp.pi * normalized_basis / phase_scale).evalf(), dtype=float)


def enumerate_lattice_points(lattice_basis, coefficient_radius):
	points = []
	for coefficient in itertools.product(range(-coefficient_radius, coefficient_radius + 1), repeat=3):
		if coefficient == (0, 0, 0):
			continue
		points.append(lattice_basis @ np.asarray(coefficient, dtype=float))
	return np.asarray(points, dtype=float)


def _voronoi_halfspaces(lattice_points):
	return np.array(
		[np.append(point, -0.5 * float(np.dot(point, point))) for point in np.asarray(lattice_points, dtype=float)],
		dtype=float,
	)


def _same_point_set(points_a, points_b, tol=1e-7):
	if len(points_a) != len(points_b):
		return False
	return all(any(np.linalg.norm(point - other) <= tol for other in points_b) for point in points_a)


def injectivity_polytope_vertices(integer_matrix, phase_scale=1.0, start_radius=1, max_radius=6, stability_tol=1e-7):
	"""
	Compute the actual injectivity region for the minimum-||v|| selection rule.

	If the decoder selects, among all v satisfying wrap(A v) = phi, the one with
	smallest Euclidean norm, then the recovered region is exactly the Euclidean
	Voronoi cell of the ambiguity lattice around the origin.
	"""
	lattice_basis = integer_ambiguity_basis(integer_matrix, phase_scale=phase_scale)
	previous_vertices = None
	previous_lattice_points = None

	for coefficient_radius in range(start_radius, max_radius + 1):
		lattice_points = enumerate_lattice_points(lattice_basis, coefficient_radius)
		halfspaces = _voronoi_halfspaces(lattice_points)
		intersection = HalfspaceIntersection(halfspaces, np.zeros(3, dtype=float))
		vertices = np.unique(np.round(intersection.intersections, decimals=8), axis=0)

		if previous_vertices is not None and _same_point_set(vertices, previous_vertices, tol=stability_tol):
			return vertices, lattice_basis, lattice_points

		previous_vertices = vertices
		previous_lattice_points = lattice_points

	if previous_vertices is None:
		raise ValueError("Failed to construct the injectivity polytope.")

	return previous_vertices, lattice_basis, previous_lattice_points


def injectivity_polytope_hull(integer_matrix, phase_scale=1.0, **kwargs):
	vertices, lattice_basis, lattice_points = injectivity_polytope_vertices(
		integer_matrix,
		phase_scale=phase_scale,
		**kwargs,
	)
	return vertices, lattice_basis, lattice_points, ConvexHull(vertices)


def point_is_in_hull(hull, point, tol=1e-8):
	point = np.asarray(point, dtype=float)
	augmented = np.append(point, 1.0)
	return np.all(hull.equations @ augmented <= tol)


def point_is_in_principal_branch(matrix, velocity, wrap_limit=np.pi, atol=1e-8):
	matrix = _validate_rank3_matrix(matrix)
	velocity = np.asarray(velocity, dtype=float)
	return np.all(np.abs(matrix @ velocity) <= wrap_limit + atol)


def extract_polytope_faces(vertices, hull, plane_tol=1e-8, incidence_tol=1e-7):
	"""
	Merge coplanar hull simplices into actual polygon faces.

	`scipy.spatial.ConvexHull` triangulates non-triangular faces. For plotting the
	actual polytope, we group simplices by supporting plane and recover each face
	as an ordered polygon.
	"""
	vertices = np.asarray(vertices, dtype=float)
	plane_groups = {}

	for equation in hull.equations:
		normal = np.asarray(equation[:-1], dtype=float)
		offset = float(equation[-1])
		normal = normal / np.linalg.norm(normal)
		pivot = int(np.argmax(np.abs(normal)))
		if normal[pivot] < 0:
			normal = -normal
			offset = -offset
		key = tuple(np.round(np.append(normal, offset), decimals=int(abs(math.log10(plane_tol)))))
		plane_groups[key] = (normal, offset)

	faces = []
	for normal, offset in plane_groups.values():
		indices = [index for index, vertex in enumerate(vertices) if abs(np.dot(normal, vertex) + offset) <= incidence_tol]
		face_vertices = vertices[indices]
		center = np.mean(face_vertices, axis=0)

		# Build an in-plane basis for angular ordering.
		trial_axis = np.array([1.0, 0.0, 0.0])
		if abs(np.dot(trial_axis, normal)) > 0.9:
			trial_axis = np.array([0.0, 1.0, 0.0])
		axis_u = trial_axis - np.dot(trial_axis, normal) * normal
		axis_u /= np.linalg.norm(axis_u)
		axis_v = np.cross(normal, axis_u)

		angles = []
		for vertex in face_vertices:
			relative = vertex - center
			angles.append(math.atan2(np.dot(relative, axis_v), np.dot(relative, axis_u)))

		ordered = [vertex for _, vertex in sorted(zip(angles, face_vertices), key=lambda pair: pair[0])]
		faces.append(np.asarray(ordered, dtype=float))

	return faces


def plot_polytope(vertices, hull, title):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection

	figure = plt.figure()
	axis = figure.add_subplot(111, projection="3d")
	axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="crimson", s=30)

	for polygon in extract_polytope_faces(vertices, hull):
		face = Poly3DCollection([polygon], alpha=0.2, facecolor="cyan", edgecolor="black", linewidths=1.0)
		axis.add_collection3d(face)

	span = np.ptp(vertices, axis=0).max() / 2.0
	center = np.mean(vertices, axis=0)
	axis.set_xlim(center[0] - span, center[0] + span)
	axis.set_ylim(center[1] - span, center[1] + span)
	axis.set_zlim(center[2] - span, center[2] + span)
	axis.set_xlabel("Vx")
	axis.set_ylabel("Vy")
	axis.set_zlabel("Vz")
	axis.set_title(title)
	plt.show()


def plot_recovery_cells(cells, title):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection

	figure = plt.figure()
	axis = figure.add_subplot(111, projection="3d")

	all_vertices = np.vstack([cell["vertices"] for cell in cells])
	axis.scatter(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2], color="crimson", s=20)

	for cell in cells:
		for polygon in extract_polytope_faces(cell["vertices"], cell["hull"]):
			face = Poly3DCollection([polygon], alpha=0.18, facecolor="cyan", edgecolor="black", linewidths=0.8)
			axis.add_collection3d(face)

	span = np.ptp(all_vertices, axis=0).max() / 2.0
	center = np.mean(all_vertices, axis=0)
	axis.set_xlim(center[0] - span, center[0] + span)
	axis.set_ylim(center[1] - span, center[1] + span)
	axis.set_zlim(center[2] - span, center[2] + span)
	axis.set_xlabel("Vx")
	axis.set_ylabel("Vy")
	axis.set_zlabel("Vz")
	axis.set_title(title)
	plt.show()


def test_tetrahedral_principal_branch_has_six_axis_vertices():
	scaled_matrix = build_scaled_matrix(EXAMPLE_A_INT, EXAMPLE_PHASE_SCALE)
	vertices = principal_branch_polytope_vertices(scaled_matrix)
	axis_extent = EXAMPLE_VENC * math.sqrt(3.0)
	expected = []

	for axis in range(3):
		for sign in (-1.0, 1.0):
			vertex = np.zeros(3, dtype=float)
			vertex[axis] = sign * axis_extent
			expected.append(vertex)

	assert len(vertices) == 6
	for vertex in expected:
		assert any(np.allclose(vertex, candidate, atol=1e-8) for candidate in vertices)


def test_tetrahedral_injectivity_polytope_has_fourteen_vertices():
	vertices, _, _ = injectivity_polytope_vertices(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	assert len(vertices) == 14


def test_tetrahedral_injectivity_polytope_has_twelve_rhombic_faces():
	vertices, _, _, hull = injectivity_polytope_hull(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	faces = extract_polytope_faces(vertices, hull)
	assert len(faces) == 12
	assert all(len(face) == 4 for face in faces)


def test_dependent_row_can_expand_injectivity_without_changing_principal_branch_extent():
	base_matrix = np.array([[7, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
	augmented_matrix = np.array([[7, 0, 0], [3, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

	base_principal = principal_branch_polytope_vertices(build_scaled_matrix(base_matrix, 1.0))
	augmented_principal = principal_branch_polytope_vertices(build_scaled_matrix(augmented_matrix, 1.0))
	base_injective, _, _ = injectivity_polytope_vertices(base_matrix, phase_scale=1.0, max_radius=3)
	augmented_injective, _, _ = injectivity_polytope_vertices(augmented_matrix, phase_scale=1.0, max_radius=3)

	assert np.isclose(np.max(np.abs(base_principal[:, 0])), np.pi / 7.0, atol=1e-8)
	assert np.isclose(np.max(np.abs(augmented_principal[:, 0])), np.pi / 7.0, atol=1e-8)
	assert np.isclose(np.max(np.abs(base_injective[:, 0])), np.pi / 7.0, atol=1e-8)
	assert np.isclose(np.max(np.abs(augmented_injective[:, 0])), np.pi, atol=1e-8)


def test_injectivity_vertices_leave_cell_under_lattice_translate():
	scaled_matrix = build_scaled_matrix(EXAMPLE_A_INT, EXAMPLE_PHASE_SCALE)
	vertices, _, lattice_points, hull = injectivity_polytope_hull(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	rng = np.random.default_rng(0)

	for _ in range(20):
		weights = rng.random(len(vertices))
		weights /= weights.sum()
		point = weights @ vertices
		assert point_is_in_hull(hull, point)
		for ambiguity in lattice_points[:12]:
			assert wrapped_encodings_match(scaled_matrix, point, point + ambiguity)
			assert not point_is_in_hull(hull, point + ambiguity)


def test_minimum_norm_representative_recovers_points_inside_injectivity_cell():
	vertices, lattice_basis, _ = injectivity_polytope_vertices(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	rng = np.random.default_rng(1)

	for _ in range(20):
		weights = rng.random(len(vertices))
		weights /= weights.sum()
		point = weights @ vertices
		representative = minimum_norm_representative(point, lattice_basis, coefficient_radius=3)
		assert np.allclose(representative, point, atol=1e-8)

		translated = point + lattice_basis[:, 0]
		representative = minimum_norm_representative(translated, lattice_basis, coefficient_radius=3)
		assert np.allclose(representative, point, atol=1e-8)


def test_bruteforce_decoder_matches_injectivity_cell_membership():
	scaled_matrix = build_scaled_matrix(EXAMPLE_A_INT, EXAMPLE_PHASE_SCALE)
	vertices, _, _, hull = injectivity_polytope_hull(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	rng = np.random.default_rng(4)

	for _ in range(60):
		point = rng.uniform(-180.0, 180.0, size=3)
		wrapped_phase = wrap(scaled_matrix @ point)
		decoded = decode_minimum_norm_from_phase(scaled_matrix, wrapped_phase, wrap_radius=3)
		decodes_to_self = np.linalg.norm(decoded - point) < 1e-5
		assert decodes_to_self == point_is_in_hull(hull, point, tol=1e-7)


def test_phase_scale_rescales_injectivity_polytope():
	normalized_vertices, _, _ = injectivity_polytope_vertices(EXAMPLE_A_INT, phase_scale=1.0, max_radius=4)
	scaled_vertices, _, _ = injectivity_polytope_vertices(EXAMPLE_A_INT, phase_scale=EXAMPLE_PHASE_SCALE, max_radius=4)
	assert np.allclose(scaled_vertices, normalized_vertices / EXAMPLE_PHASE_SCALE, atol=1e-8)


def test_pseudoinverse_exact_recovery_region_can_stay_local_with_dependent_rows():
	base_matrix = build_scaled_matrix(np.array([[7, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int), 1.0)
	augmented_matrix = build_scaled_matrix(np.array([[7, 0, 0], [3, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int), 1.0)

	base_cells = exact_recovery_cells(base_matrix, make_pseudoinverse_decoder(base_matrix), wrap_radius=2)
	augmented_cells = exact_recovery_cells(augmented_matrix, make_pseudoinverse_decoder(augmented_matrix), wrap_radius=7)

	base_vertices = np.vstack([cell["vertices"] for cell in base_cells])
	augmented_vertices = np.vstack([cell["vertices"] for cell in augmented_cells])
	assert np.isclose(np.max(np.abs(base_vertices[:, 0])), np.pi / 7.0, atol=1e-8)
	assert np.isclose(np.max(np.abs(augmented_vertices[:, 0])), np.pi / 7.0, atol=1e-8)


def test_five_point_decoder_exact_recovery_matches_original_predicate_on_samples():
	full_matrix = build_scaled_matrix(EXAMPLE_A0_INT, EXAMPLE_PHASE_SCALE)
	decoder = make_five_point_mri_decoder(full_matrix)
	cells = exact_recovery_cells(full_matrix, decoder, wrap_radius=2)
	assert len(cells) == 1
	assert np.array_equal(cells[0]["wrap_index"], np.zeros(full_matrix.shape[0], dtype=int))

	measured_matrix = full_matrix[:-1, :]
	full_pseudoinverse = np.linalg.pinv(full_matrix)
	measured_pseudoinverse = np.linalg.pinv(measured_matrix)

	def is_velocity_resolvable(v):
		phi_0 = wrap(full_matrix @ v)
		phi_1 = phi_0[:-1]
		vsolve = measured_pseudoinverse @ (
			phi_1 + 2.0 * np.pi * np.round(measured_matrix @ full_pseudoinverse @ phi_0 - phi_1) / (2.0 * np.pi)
		)
		return np.all(np.abs(v - vsolve) < 1e-5)

	rng = np.random.default_rng(2)
	for _ in range(400):
		point = rng.uniform(-160.0, 160.0, size=3)
		assert is_velocity_resolvable(point) == point_is_in_exact_recovery_cells(full_matrix, point, cells)


def test_general_difference_decoder_reproduces_five_point_decoder():
	full_matrix = build_scaled_matrix(EXAMPLE_A0_INT, EXAMPLE_PHASE_SCALE)
	reference_decoder = make_five_point_mri_decoder(full_matrix)
	general_decoder = make_difference_based_mri_decoder(full_matrix)
	rng = np.random.default_rng(3)

	for _ in range(300):
		point = rng.uniform(-160.0, 160.0, size=3)
		wrapped_phase = wrap(full_matrix @ point)
		assert np.allclose(reference_decoder(wrapped_phase), general_decoder(wrapped_phase), atol=1e-10)


if __name__ == "__main__":
	full_matrix = build_scaled_matrix(EXAMPLE_A0_INT, EXAMPLE_PHASE_SCALE)
	decoder = make_difference_based_mri_decoder(full_matrix)
	cells = exact_recovery_cells(full_matrix, decoder, wrap_radius=2)
	print("Difference-based MRI exact-recovery cell count:", len(cells))
	for cell in cells:
		print("wrap index:", cell["wrap_index"].tolist(), "vertex count:", len(cell["vertices"]))
	plot_recovery_cells(cells, "Difference-based MRI exact-recovery region")
