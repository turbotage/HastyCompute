
import numpy as np
import matplotlib.pyplot as plt

def view_over_time():
	thresh = "0.0006"
	#extra = "_no_weights"
	extra = ""

	from_zero = np.load(f"D:/4DRecon/results/from_zero_{thresh}{extra}.npy")
	from_mean = np.load(f"D:/4DRecon/results/from_mean_{thresh}{extra}.npy")
	from_difference = np.load(f"D:/4DRecon/results/from_difference_{thresh}{extra}.npy")

	plt.figure()
	plt.plot(from_zero, 'r-*')
	plt.plot(from_mean, 'g-*')
	plt.plot(from_difference, 'b-*')
	plt.legend(['from_zero', 'from_mean', 'from_differenc'])
	plt.show()

	plt.figure()
	plt.plot(np.log(from_zero), 'r-*')
	plt.plot(np.log(from_mean), 'g-*')
	plt.plot(np.log(from_difference), 'b-*')
	plt.legend(['from_zero', 'from_mean', 'from_differenc'])
	plt.show()

	print('Herre')

def view_over_thresh():
	#threshes = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
	#threshes = [1e-7, 5e-8, 1e-8]
	#threshes = [5e-9, 1e-9, 5e-10]
	threshes = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 5e-10, 1e-10, 5e-11, 1e-11]
	from_zeros = []
	from_means = []
	from_diffs = []

	for thresh in threshes:
		from_zeros.append(np.load(f"D:/4DRecon/results/from_zero_{thresh}.npy")[-1])
		from_means.append(np.load(f"D:/4DRecon/results/from_mean_{thresh}.npy")[-1])
		from_diffs.append(np.load(f"D:/4DRecon/results/from_difference_{thresh}.npy")[-1])

	#x = np.log(np.array(threshes))
	x = np.array(threshes)

	plt.figure()
	#plt.plot(x, np.array(from_zeros), 'r-*')
	plt.plot(x, np.array(from_means), 'g-*')
	plt.plot(x, np.array(from_diffs), 'b-*')
	#plt.legend(['from_zero', 'from_mean', 'from_differenc'])
	plt.show()

view_over_thresh()