import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import Markov


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

# Create 6000 x 1000 location matrix from Labels.csv
# M[i][j] = (x, y) --> At Run i + 1 and Step j + 1, bot is a location (x, y)
def createLocationMatrix():
	labels = np.genfromtxt("../Label.csv", delimiter=',')
	M = np.zeros((6000, 1000), dtype = 'float32, float32')

	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		M[int(run-1), int(step-1)] = (x, y)

	return M

# Create observations matrix from Observations.csv
# M[i][j] = Angle of bot to x-axis on Run i + 1 and Step j + 1
def createObservationMatrix():
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')

	return observations 

def guassianHMM():
	model = hmm.GaussianHMM(n_components=3, covariance_type="full")

def createSubmission(matrix):
	N = np.zeros((12000, 1))

	for i in range(matrix.shape[0]):
		N[2*i, :] = matrix[i, 0]
		N[2*i+1, :] = matrix[i, 1]

	np.savetxt('hmm_submission.csv', N, fmt='%f', delimiter=",", header="Id,Label", comments='')


def run():
	LM = createLocationMatrix()
	OM = createObservationMatrix()


if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()

