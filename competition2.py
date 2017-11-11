import numpy as np
import matplotlib.pyplot as plt

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
		M[row, column] = (x, y)

	return M


def run():
	M = createLocationMatrix(labels)




if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()
	runVisuals()