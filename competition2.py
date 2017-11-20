import numpy as np
import matplotlib.pyplot as plt

print 'Loading files...'
observations = np.genfromtxt("../data/Observations.csv", delimiter = ',')
labels = np.genfromtxt("../data/Label.csv", delimiter=',')
print 'Done loading files'

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
	print 'Creating location matrix...'
	M = np.zeros((6000, 1000), dtype = 'float32, float32')

	stepDict = {}

	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		run = int(run)
		step = int(step)
		if step not in stepDict:
			stepDict[step] = {
				'values': set(),
				'count': 0
			}
		stepDict[step]['values'].add((x,y))
		stepDict[step]['count'] += 1
		M[run-1, step-1] = (x, y)

	print 'Done creating location matrix'
	for step, value in stepDict.items():
		print 'Step: ', step, '|', len(value['values']), '/', value['count'], 'unique locs'
	return M


def run():
	M = createLocationMatrix(labels)


if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()
	# runVisuals()