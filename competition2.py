import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import Markov


observations = np.genfromtxt("Observations.csv", delimiter = ',')
labels = np.genfromtxt("Label.csv", delimiter=',')


# Visualize the clustering on a 2D graph
# Credit: Special thanks to Ilan Filonenko for this plotting approach
def visualizeClusters(M, clusters, title="Clustering"):
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
	tsne_data = tsne.fit_transform(np.squeeze(np.asarray(M)))

	plt.title(title)
	plt.scatter(tsne_data[:,0], tsne_data[:,1], c=clusters, cmap=plt.cm.get_cmap("jet", 10))
	plt.colorbar(ticks=range(10))
	plt.clim(-0.5, 9.5)
	plt.colorbar()
	plt.show()

def createLocationMatrix(labels):
	M = np.zeros((6000, 1000), dtype = 'float32, float32')

	for i in range(labels.shape[0]):
		row, column, x, y = labels[i, :]
		M[int(row), int(column)] = (x, y)

	return M

def createObservationMatrix(observations):
	O = np.zeros((10000, 1000))
	for i in range(observations.shape[0]):
		for j in range(observations.shape[1]):
			O[i, j] = observations[i, j]

	return O

def guassianHMM():
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")

def haha():
	return

def createSubmission(matrix):
	N = np.zeros((12000, 1))

	for i in range(matrix.shape[0]):
		N[2i, :] = matrix[i, 0]
		N[2i+1, :] = matrix[i, 1]

	np.savetxt('hmm_submission.csv', N, fmt='%f', delimiter=",", header="Id,Label", comments='')


def run():
	M = createLocationMatrix(labels)
	O = createObservationMatrix(observations)




if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()
	runVisuals()