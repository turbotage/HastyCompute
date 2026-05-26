import argparse
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


DEFAULT_VENC = 35.0
C1 = 2
C2 = 1
DEFAULT_A_INT = np.array(
	[
		[-C2, -C2, -C2],
		[C2, C2, -C2],
		[C2, -C2, C2],
		[-C2, C2, C2],
		[-C1, -C1, -C1],
		[C1, C1, -C1],
		[C1, -C1, C1],
		[-C1, C1, C1],
		[0, 0, 0],
	],
	dtype=np.float32,
)


def wrap_torch(phases: torch.Tensor) -> torch.Tensor:
	return torch.remainder(phases + math.pi, 2.0 * math.pi) - math.pi


def build_scaled_matrix(integer_matrix: np.ndarray, venc: float) -> np.ndarray:
	return (math.pi / (math.sqrt(3.0) * venc)) * np.asarray(integer_matrix, dtype=np.float32)


def default_difference_decoder_rows(full_matrix: torch.Tensor, row_tol: float = 1e-12) -> list[int]:
	row_norms = torch.linalg.norm(full_matrix, dim=1)
	solve_rows = [index for index, norm in enumerate(row_norms.tolist()) if norm > row_tol]
	if not solve_rows:
		raise ValueError("No non-zero rows are available for reconstruction.")
	if torch.linalg.matrix_rank(full_matrix[solve_rows, :]).item() < 3:
		raise ValueError("The non-zero rows do not span a rank-3 reconstruction system.")
	return solve_rows


def make_difference_based_mri_decoder_torch(
	full_matrix: torch.Tensor,
	solve_rows: list[int] | None = None,
	estimate_rows: list[int] | None = None,
	row_tol: float = 1e-12,
):
	if full_matrix.ndim != 2 or full_matrix.shape[1] != 3:
		raise ValueError("full_matrix must have shape [m, 3].")
	if torch.linalg.matrix_rank(full_matrix).item() < 3:
		raise ValueError("full_matrix must have rank 3.")

	if solve_rows is None:
		solve_rows = default_difference_decoder_rows(full_matrix, row_tol=row_tol)
	if estimate_rows is None:
		estimate_rows = list(range(full_matrix.shape[0]))

	solve_matrix = full_matrix[solve_rows, :]
	estimate_matrix = full_matrix[estimate_rows, :]
	if torch.linalg.matrix_rank(solve_matrix).item() < 3:
		raise ValueError("solve_rows must define a rank-3 reconstruction matrix.")
	if torch.linalg.matrix_rank(estimate_matrix).item() < 3:
		raise ValueError("estimate_rows must define a rank-3 estimation matrix.")

	solve_pseudoinverse = torch.linalg.pinv(solve_matrix)
	estimate_pseudoinverse = torch.linalg.pinv(estimate_matrix)
	estimate_to_solve = solve_matrix @ estimate_pseudoinverse

	def decoder(wrapped_phase: torch.Tensor) -> torch.Tensor:
		estimate_phase = wrapped_phase[:, estimate_rows]
		solve_phase = wrapped_phase[:, solve_rows]
		wrap_counts = torch.round((estimate_phase @ estimate_to_solve.T - solve_phase) / (2.0 * math.pi))
		return (solve_phase + 2.0 * math.pi * wrap_counts) @ solve_pseudoinverse.T

	return decoder


def make_five_point_mri_decoder_torch(full_matrix: torch.Tensor):
	return make_difference_based_mri_decoder_torch(
		full_matrix,
		solve_rows=list(range(full_matrix.shape[0] - 1)),
		estimate_rows=list(range(full_matrix.shape[0])),
	)


def generate_edge_points(venc: float, device: torch.device, dtype: torch.dtype, eps: float = 1e-5) -> torch.Tensor:
	venc_eps = venc - eps
	patterns: list[list[float]] = []

	def append_patterns(active_axes: int, amplitude: float):
		for axes in itertools.combinations(range(3), active_axes):
			for signs in itertools.product((-1.0, 1.0), repeat=active_axes):
				point = [0.0, 0.0, 0.0]
				for axis, sign in zip(axes, signs):
					point[axis] = sign * amplitude
				patterns.append(point)

	append_patterns(1, venc_eps * math.sqrt(3.0))
	append_patterns(2, venc_eps * math.sqrt(2.0))
	append_patterns(3, venc_eps)
	append_patterns(1, venc_eps)
	append_patterns(2, venc_eps / math.sqrt(2.0))
	append_patterns(3, venc_eps / math.sqrt(3.0))
	return torch.tensor(patterns, device=device, dtype=dtype)


def generate_candidate_points(
	venc: float,
	vlim: float,
	n_points_per_axis: int,
	device: torch.device,
	dtype: torch.dtype,
	min_radius: float | None,
) -> torch.Tensor:
	axis = torch.linspace(-vlim, vlim, n_points_per_axis, device=device, dtype=dtype)
	x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
	points = torch.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), dim=1)
	norms = torch.linalg.norm(points, dim=1)
	if min_radius is None:
		min_radius = 0.0
	mask = (norms >= float(min_radius)) & (norms <= float(vlim))
	points = points[mask]
	return torch.cat((points, generate_edge_points(venc, device=device, dtype=dtype)), dim=0)


@torch.no_grad()
def sample_resolvable_points(
	encoding_matrix: torch.Tensor,
	decoder,
	venc: float,
	vlim: float,
	n_points_per_axis: int,
	batch_size: int,
	atol: float,
	min_radius: float | None,
) -> np.ndarray:
	points = generate_candidate_points(
		venc=venc,
		vlim=vlim,
		n_points_per_axis=n_points_per_axis,
		device=encoding_matrix.device,
		dtype=encoding_matrix.dtype,
		min_radius=min_radius,
	)
	resolvable_batches = []

	for start in range(0, points.shape[0], batch_size):
		batch = points[start : start + batch_size]
		wrapped_phase = wrap_torch(batch @ encoding_matrix.T)
		decoded = decoder(wrapped_phase)
		mask = torch.all(torch.abs(decoded - batch) <= atol, dim=1)
		if torch.any(mask):
			resolvable_batches.append(batch[mask].detach().cpu())

	if not resolvable_batches:
		return np.empty((0, 3), dtype=np.float32)

	return torch.cat(resolvable_batches, dim=0).numpy()


def compute_convex_hull(points: np.ndarray) -> ConvexHull | None:
	if len(points) < 4:
		return None
	try:
		return ConvexHull(points)
	except Exception:
		return None


def plot_hull(points: np.ndarray, hull: ConvexHull, title: str) -> None:
	figure = plt.figure()
	axis = figure.add_subplot(111, projection="3d")
	axis.scatter(points[:, 0], points[:, 1], points[:, 2], color="crimson", s=8)

	for simplex in hull.simplices:
		triangle = points[simplex]
		face = Poly3DCollection([triangle], alpha=0.18, facecolor="cyan", edgecolor="black", linewidths=0.8)
		axis.add_collection3d(face)

	span = np.ptp(points, axis=0).max() / 2.0
	center = np.mean(points, axis=0)
	axis.set_xlim(center[0] - span, center[0] + span)
	axis.set_ylim(center[1] - span, center[1] + span)
	axis.set_zlim(center[2] - span, center[2] + span)
	axis.set_xlabel("Vx")
	axis.set_ylabel("Vy")
	axis.set_zlabel("Vz")
	axis.set_title(title)
	plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="CUDA brute-force exact-recovery sampler for venc decoders.")
	parser.add_argument("--venc", type=float, default=DEFAULT_VENC)
	parser.add_argument("--vlim", type=float, default=None)
	parser.add_argument("--n-points-per-axis", type=int, default=96)
	parser.add_argument("--batch-size", type=int, default=262144)
	parser.add_argument("--atol", type=float, default=1e-4)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=("float32", "float64"), default="float32")
	parser.add_argument("--min-radius", type=float, default=None, help="Minimum |v| to sample. Default samples the full ball from the origin.")
	parser.add_argument("--decoder", type=str, choices=("difference", "five-point"), default="difference")
	parser.add_argument("--no-plot", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available.")

	device = torch.device(args.device)
	dtype = getattr(torch, args.dtype)
	vlim = float(args.vlim if args.vlim is not None else args.venc)

	full_matrix_np = build_scaled_matrix(DEFAULT_A_INT, args.venc)
	full_matrix = torch.tensor(full_matrix_np, device=device, dtype=dtype)

	if args.decoder == "difference":
		decoder = make_difference_based_mri_decoder_torch(full_matrix)
		title = "Difference-based MRI exact-recovery hull"
	else:
		decoder = make_five_point_mri_decoder_torch(full_matrix)
		title = "Five-point MRI exact-recovery hull"

	resolvable_points = sample_resolvable_points(
		encoding_matrix=full_matrix,
		decoder=decoder,
		venc=args.venc,
		vlim=vlim,
		n_points_per_axis=args.n_points_per_axis,
		batch_size=args.batch_size,
		atol=args.atol,
		min_radius=args.min_radius,
	)
	print(f"Sampled resolvable points: {len(resolvable_points)}")
	if len(resolvable_points) == 0:
		print("No resolvable points were found with the current sampling settings.")
		return

	hull = compute_convex_hull(resolvable_points)
	if hull is None:
		print("Not enough points for a convex hull.")
		return

	print(f"Hull vertices: {len(hull.vertices)}")
	print(f"Hull simplices: {len(hull.simplices)}")
	print(f"Min/Max sampled |v|: {np.linalg.norm(resolvable_points, axis=1).min():.3f} / {np.linalg.norm(resolvable_points, axis=1).max():.3f}")

	if not args.no_plot:
		plot_hull(resolvable_points, hull, title)


if __name__ == "__main__":
	main()