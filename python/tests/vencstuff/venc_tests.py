import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

def compute_solvable_parallelepiped(is_solved, venc, vlim=50, n_points_per_axis=20):
	"""
	Sample 3D velocities and collect points that are solved.

	is_solved : callable
		Function taking a (3,) velocity and returning True if solved.
	vlim : float
		Maximum velocity magnitude to sample along each axis.
	n_points_per_axis : int
		Number of points per axis in grid.

	Returns: numpy array of solvable points
	"""
	axis = np.linspace(-vlim, vlim, n_points_per_axis)
	X, Y, Z = np.meshgrid(axis, axis, axis)
	points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
	point_mags = np.linalg.norm(points, axis=1)
	points = points[(point_mags >= venc) & (point_mags <= vlim)]

	venc_eps = venc - 1e-5

	# add the edge points on the sphere
	edge_points = np.array([
							[venc_eps*math.sqrt(3), 0, 0],
							[0, venc_eps*math.sqrt(3), 0],
							[0, 0, venc_eps*math.sqrt(3)],
							[-venc_eps*math.sqrt(3), 0, 0],
							[0, -venc_eps*math.sqrt(3), 0],
							[0, 0, -venc_eps*math.sqrt(3)],
							[venc_eps*math.sqrt(2), venc_eps*math.sqrt(2), 0],
							[venc_eps*math.sqrt(2), 0, venc_eps*math.sqrt(2)],
							[0, venc_eps*math.sqrt(2), venc_eps*math.sqrt(2)],
							[-venc_eps*math.sqrt(2), venc_eps*math.sqrt(2), 0],
							[-venc_eps*math.sqrt(2), 0, venc_eps*math.sqrt(2)],
							[0, -venc_eps*math.sqrt(2), venc_eps*math.sqrt(2)],
							[venc_eps*math.sqrt(2), -venc_eps*math.sqrt(2), 0],
							[venc_eps*math.sqrt(2), 0, -venc_eps*math.sqrt(2)],
							[0, venc_eps*math.sqrt(2), -venc_eps*math.sqrt(2)],
							[-venc_eps*math.sqrt(2), -venc_eps*math.sqrt(2), 0],
							[-venc_eps*math.sqrt(2), 0, -venc_eps*math.sqrt(2)],
							[0, -venc_eps*math.sqrt(2), -venc_eps*math.sqrt(2)],
							[venc_eps, venc_eps, venc_eps],
							[-venc_eps, venc_eps, venc_eps],
							[venc_eps, -venc_eps, venc_eps],
							[venc_eps, venc_eps, -venc_eps],
							[-venc_eps, -venc_eps, venc_eps],
							[-venc_eps, venc_eps, -venc_eps],
							[venc_eps, -venc_eps, -venc_eps],
							[-venc_eps, -venc_eps, -venc_eps],
							[venc_eps, 0, 0],
							[0, venc_eps, 0],
							[0, 0, venc_eps],
							[-venc_eps, 0, 0],
							[0, -venc_eps, 0],
							[0, 0, -venc_eps],
							[venc_eps/math.sqrt(2), venc_eps/math.sqrt(2), 0],
							[venc_eps/math.sqrt(2), 0, venc_eps/math.sqrt(2)],
							[0, venc_eps/math.sqrt(2), venc_eps/math.sqrt(2)],
							[-venc_eps/math.sqrt(2), venc_eps/math.sqrt(2), 0],
							[-venc_eps/math.sqrt(2), 0, venc_eps/math.sqrt(2)],
							[0, -venc_eps/math.sqrt(2), venc_eps/math.sqrt(2)],
							[venc_eps/math.sqrt(2), -venc_eps/math.sqrt(2), 0],
							[venc_eps/math.sqrt(2), 0, -venc_eps/math.sqrt(2)],
							[0, venc_eps/math.sqrt(2), -venc_eps/math.sqrt(2)],
							[-venc_eps/math.sqrt(2), -venc_eps/math.sqrt(2), 0],
							[-venc_eps/math.sqrt(2), 0, -venc_eps/math.sqrt(2)],
							[0, -venc_eps/math.sqrt(2), -venc_eps/math.sqrt(2)],
							[venc_eps/math.sqrt(3), venc_eps/math.sqrt(3), venc_eps/math.sqrt(3)],
							[-venc_eps/math.sqrt(3), venc_eps/math.sqrt(3), venc_eps/math.sqrt(3)],
							[venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3), venc_eps/math.sqrt(3)],
							[venc_eps/math.sqrt(3), venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3)],
							[-venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3), venc_eps/math.sqrt(3)],
							[-venc_eps/math.sqrt(3), venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3)],
							[venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3)],
							[-venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3), -venc_eps/math.sqrt(3)]
	])

	points = np.vstack([points, edge_points])

	solvable_points = []
	for v in points:
		if is_solved(v):
			solvable_points.append(v)
	
	return np.array(solvable_points)

def plot_parallelepiped(points, alpha=0.2):
	"""
	Plot a 3D convex hull of points as a translucent parallelepiped.
	"""
	hull = ConvexHull(points)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# faces
	for simplex in hull.simplices:
		tri = points[simplex]
		poly = Poly3DCollection([tri], alpha=alpha, facecolor='cyan')
		ax.add_collection3d(poly)

	# edges
	for simplex in hull.simplices:
		for i in range(3):
			start = points[simplex[i]]
			end = points[simplex[(i+1)%3]]
			ax.plot(*zip(start, end), color='blue', linewidth=1)

	#ax.scatter(points[:,0], points[:,1], points[:,2], color='red', s=5)

	# equal aspect
	max_range = np.ptp(points, axis=0).max()
	mid = np.mean(points, axis=0)
	ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
	ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
	ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)

	ax.set_xlabel('Vx')
	ax.set_ylabel('Vy')
	ax.set_zlabel('Vz')
	ax.set_title('Resolvable velocity space')

	plt.show()

	return hull

def hull_statistics(hull, vertices, max_samples=10000):

	areas = []

	for simplex in hull.simplices:
		tri = vertices[simplex]  # 3x3 array
		# Compute area of the triangle
		a = tri[1] - tri[0]
		b = tri[2] - tri[0]
		area = 0.5 * np.linalg.norm(np.cross(a, b))
		areas.append(area)

	areas = np.array(areas)

	def average_magnitude_on_triangle(tri, n_samples):
		# tri is 3x3 array of triangle vertices
		r1 = np.sqrt(np.random.rand(n_samples))
		r2 = np.random.rand(n_samples)
		u = 1 - r1
		v = r1 * (1 - r2)
		w = r1 * r2
		points = u[:,None]*tri[0] + v[:,None]*tri[1] + w[:,None]*tri[2]
		return points

	max_area = np.max(areas)

	all_mags = []
	for i, simplex in enumerate(hull.simplices):
		tri = vertices[simplex]
		sample_points = average_magnitude_on_triangle(tri, int(areas[i] / max_area * max_samples))
		mags = np.linalg.norm(sample_points, axis=1)
		all_mags.append(mags)

	all_mags = np.concatenate(all_mags)
	
	plt.figure()
	plt.hist(all_mags, bins=50)
	plt.title("Histogram of resolvable velocity magnitudes")
	plt.xlabel("Velocity magnitude")
	plt.ylabel("Counts")
	plt.show()

	mean_mag = np.mean(all_mags)
	std_mag = np.std(all_mags)
	meadian_mag = np.median(all_mags)
	min_mag = np.min(all_mags)
	max_mag = np.max(all_mags)

	return mean_mag, std_mag, meadian_mag, min_mag, max_mag

E = np.array([
	[-1, -1, -1],
	[ 1,  1, -1],
	[ 1, -1,  1],
	[-1,  1,  1],
	[ 0,  0,  0]
])
venc = 70.0

A0 = (np.pi / (math.sqrt(3) * venc)) * E
A1 = A0[:-1,:]
A0p = np.linalg.pinv(A0)
A1p = np.linalg.pinv(A1)
def is_velocity_resolvable(v):
	phi_0 = A0 @ v
	phi_0 = np.mod(phi_0 + np.pi, 2*np.pi) - np.pi
	phi_1 = phi_0[:-1]
	
	vsolve = A1p @ (phi_1 + 2*np.pi*np.round(A1 @ A0p @ phi_0 - phi_1) / (2*np.pi))
	
	return np.all(np.abs(v - vsolve) < 1e-5)
	


vertices = compute_solvable_parallelepiped(is_velocity_resolvable, venc=venc,vlim=70.0, n_points_per_axis=60)
hull = plot_parallelepiped(vertices)
	
mean_mag, std_mag, median_mag, min_mag, max_mag = hull_statistics(hull, vertices)
print(f"Mean magnitude of resolvable velocities: {mean_mag:.2f} ± {std_mag:.2f} units")
print(f"Median magnitude of resolvable velocities: {median_mag:.2f} units")
print(f"Min/Max magnitude of resolvable velocities: {min_mag:.2f} / {max_mag:.2f} units")