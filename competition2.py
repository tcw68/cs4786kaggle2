import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import Markov
from math import *
from sklearn.externals import joblib
import time

############
# PLOTTING #
############

# Plot the bot movement based on labelled data
def plotBotMovement():
	labels = np.genfromtxt("../Label.csv", delimiter=',')

	locations = {}
	xVals, yVals = [], []
	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		locations.setdefault(run, []).append((step, (x, y)))
		xVals.append(x+1.5)
		yVals.append(y+1.5)

	for key, val in locations.items():
		orderedVal = sorted(val, key=lambda x: x[0])
		locations[key] = orderedVal

	plt.plot(xVals, yVals, 'ro')
	plt.show()

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

#################
# PREPROCESSING #
#################

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

# Create the labels dictionary
# Dict format: {Run: [(Step1, (x1, y1)) ... ]}
def createLabelsDict():
	labels = np.genfromtxt("../Label.csv", delimiter=',')
	OM = createObservationMatrix()

	locations = {}
	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		run, step = int(run), int(step)
		observation = OM[run-1, step-1]
		val = (step, (x, y), observation)

		if run in locations:
			if val not in locations[run]:
				locations[run].append(val)
		else:
			locations[run] = [val]

	for key, val in locations.items():
		orderedVal = sorted(val, key=lambda x: x[0])
		locations[key] = orderedVal

	return locations

# Save a CSV file with labels sorted by run and step
def writeLabelSortedCSV():
	labels = np.genfromtxt("../Label.csv", delimiter=',')
	sortedLabels = np.zeros((600000, 4))
	a = labels[:, 0]
	b = labels[:, 1]

	indices = np.lexsort((b, a))
	for i, row in enumerate(indices):
		sortedLabels[i, :] = labels[indices[i]]

	print "saving"
	np.savetxt('../LabelSorted.csv', sortedLabels, fmt=['%i']*2 + ['%f']*2, delimiter=",")
 
# Create submission file
def createSubmission(matrix):
	N = np.zeros((12000, 1))

	for i in range(matrix.shape[0]):
		N[2*i, :] = matrix[i, 0]
		N[2*i+1, :] = matrix[i, 1]

	np.savetxt('hmm_submission.csv', N, fmt='%f', delimiter=",", header="Id,Label", comments='')

################
# CALCULATIONS #
################

# Calculate the (alpha, beta) offset values from original position
# alpha = beta = 1.5
def calculateAlphaBeta(x1, y1, x2, y2, theta1, theta2):
	# Calculate alpha
	alphaNum = (x2 * tan(theta2)) - (x1 * tan(theta1)) + y1 - y2
	alphaDen = tan(theta1) - tan(theta2)
	alpha =  alphaNum / alphaDen

	# Calculate beta
	beta = ((x1 + alpha) * (tan(theta1))) - y1

	return (alpha, beta)

# Calculate [(minX, minY), (maxX, maxY)] from labels
# [(minX, minY), (maxX, maxY)] = [(-1.3074, -1.2908), (1.328, 1.2674)]
def calculateMinMaxXY():
	labels = np.genfromtxt("../Label.csv", delimiter=',')

	minX, minY = float("inf"), float("inf")
	maxX, maxY = float("-inf"), float("-inf")
	for i in range(labels.shape[0]):
		label = labels[i]
		_, _, x, y = label
		minX, minY = min(x, minX), min(y, minY)
		maxX, maxY = max(x, maxX), max(y, maxY)

	minX, minY = round(minX, 4), round(minY, 4)
	maxX, maxY = round(maxX, 4), round(maxY, 4)

	return [(minX, minY), (maxX, maxY)]

# Use consecutive steps in runs to calculate average bot step size
# Avg step size = 0.15356177070436083
def calculateAverageStepSize():
	sortedLabels = np.genfromtxt("../LabelSorted.csv", delimiter=',')
	distances = []

	for i in range(1, sortedLabels.shape[0]):
		if sortedLabels[i-1, 0] == sortedLabels[i, 0]:
			if sortedLabels[i-1, 1] + 1 == sortedLabels[i, 1]:
				distances.append(np.linalg.norm(sortedLabels[i-1,2:]-sortedLabels[i,2:]))

	distances = np.array(distances)
	return np.mean(distances)

def run():
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')

	print "starting hmm"
	start_time = time.time()
	k = 4
	model = hmm.GaussianHMM(n_components=k, covariance_type="full")
	model.fit(observations)
	print "done running hmm"
	joblib.dump(model, "hmm"+str(k)+".pkl")
	print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	# labelsDict = createLabelsDict()
	# run()


