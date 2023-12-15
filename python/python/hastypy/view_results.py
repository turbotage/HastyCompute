
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

def add_threses(threshes, appenders):
	for app in appenders:
		if app in threshes:
			continue
		else:
			threshes.append(app)

def view_over_thresh():
	threshes = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
	add_threses(threshes, [1e-7, 5e-8, 1e-8])
	add_threses(threshes, [5e-9, 1e-9, 5e-10])
	add_threses(threshes, [1e-3, 2e-3, 4e-3, 8e-3, 1e-4, 3e-4, 5e-4, 8e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7])
	#add_threses(threshes, [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3])
	#add_threses(threshes, [1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3, 4e-3, 8e-3, 9e-3, 1e-2, 2e-2, 5e-2, 1e-1])
	
	threshes = threshes.sort()
	
	from_zeros_x = []
	from_zeros = []

	from_means_x = []
	from_means = []

	from_diffs_x = []
	from_diffs = []

	folder = "spokes_100_samp_100"

	for thresh in threshes:
		try:
			from_zeros.append(np.load(f"D:/4DRecon/results/{folder}/from_zero_{thresh}.npy")[-1])
			from_zeros_x.append(thresh)
		except:
			pass

		try:
			from_means.append(np.load(f"D:/4DRecon/results/{folder}/from_mean_{thresh}.npy")[-1])
			from_means_x.append(thresh)
		except:
			pass

		try:
			from_diffs.append(np.load(f"D:/4DRecon/results/{folder}/from_difference_{thresh}.npy")[-1])
			from_diffs_x.append(thresh)
		except:
			pass

	from_zeros_x = np.array(from_zeros_x)
	from_zeros = np.array(from_zeros)

	from_means_x = np.array(from_means_x)
	from_means = np.array(from_means)

	from_diffs_x = np.array(from_diffs_x)
	from_diffs = np.array(from_diffs)

	#x = np.log(np.array(threshes))
	x = np.array(threshes)

	plt.figure()
	plt.plot(from_zeros_x, from_zeros, 'r-*')
	plt.plot(from_means_x, from_means, 'g-*')
	plt.plot(from_diffs_x, from_diffs, 'b-*')
	#plt.legend(['from_mean', 'from_differenc'])
	plt.legend(['from_zero', 'from_mean', 'from_differenc'])
	plt.show()

	plt.figure()
	plt.plot(from_zeros_x, from_zeros, 'r-*')
	plt.plot(from_means_x, from_means, 'g-*')
	plt.plot(from_diffs_x, from_diffs, 'b-*')
	#plt.legend(['from_mean', 'from_differenc'])
	plt.legend(['from_zero', 'from_mean', 'from_differenc'])
	plt.show()


view_over_thresh()