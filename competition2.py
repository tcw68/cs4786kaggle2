import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import Markov
from math import *
from sklearn.externals import joblib
import time
import pickle

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

# Plot the states that the labels are assigned to
def plotPredictedStates(predictedStates, numStates=10):
	labels = np.genfromtxt("../Label.csv", delimiter=',')

	# Map each state to a distinct RGB color
	cmap = ['#f45342', '#f4a041', '#f4ee41', '#d3f441', '#7ff441', '#41f4df', '#41a6f4', '#5241f4', '#d641f4', '#f44176']

	xVals = [[] for _ in range(numStates)]
	yVals = [[] for _ in range(numStates)]
	for label in labels:
		run, step, x, y = label
		nextState = predictedStates[int(run) - 1, int(step) - 1]
		xVals[nextState].append(x + 1.5)
		yVals[nextState].append(y + 1.5)

	for i in range(numStates):
		plt.scatter(xVals[i], yVals[i], color=cmap[i], marker='.')

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
		x, y = x + 1.5, y + 1.5 # Add alpha/beta offsets
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
 
# Create submission file
def createSubmission(matrix):
	N = np.zeros((12000, 1))

	for i in range(matrix.shape[0]):
		N[2*i, :] = matrix[i, 0]
		N[2*i+1, :] = matrix[i, 1]

	np.savetxt('hmm_submission.csv', N, fmt='%f', delimiter=",", header="Id,Label", comments='')

#####################
# LOADING / WRITING #
#####################

# Load dictionary from pickle file
def loadDict(filename):
	with open(filename, 'r') as f:
		d = pickle.load(f)
		return d

# Write dictionary to pickle file
def writeDict(d, filename):
	with open(filename, 'w') as f:
		pickle.dump(d, f)

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

##############
# ALGORITHMS #
##############

"""
--- 1102.67181301 seconds --- for k = 4

array([[ 0.17633803,  0.3059155 ,  0.32      ,  0.19774648],
       [ 0.17445874,  0.32538437,  0.32099153,  0.17916536],
       [ 0.18196253,  0.31216259,  0.31343284,  0.19244204],
       [ 0.17637712,  0.32997881,  0.3029661 ,  0.19067797]])

--- 2073.54278898 seconds --- for k = 9

array([[ 0.14115011,  0.06049291,  0.06497386,  0.10978342,  0.12994772, 0.1120239 ,  0.10828977,  0.1523525 ,  0.12098581],
       [ 0.13137558,  0.08037094,  0.08964451,  0.10973725,  0.10973725, 0.12055642,  0.08655332,  0.15919629,  0.11282844],
       [ 0.13654096,  0.06892068,  0.08062419,  0.11183355,  0.12093628, 0.12093628,  0.11053316,  0.11313394,  0.13654096],
       [ 0.1332737 ,  0.06171735,  0.09123435,  0.10107335,  0.10822898, 0.12701252,  0.10733453,  0.14669052,  0.1234347 ],
       [ 0.14659686,  0.05235602,  0.07068063,  0.10471204,  0.12216405, 0.13176265,  0.11256544,  0.14397906,  0.11518325],
       [ 0.12286159,  0.06531882,  0.06376361,  0.12130638,  0.11197512, 0.13297045,  0.12130638,  0.15241058,  0.10808709],
       [ 0.12383613,  0.0689013 ,  0.08007449,  0.12011173,  0.10707635, 0.1471136 ,  0.10242086,  0.12383613,  0.12662942],
       [ 0.1277193 ,  0.06596491,  0.08350877,  0.11438596,  0.11649123, 0.12842105,  0.0954386 ,  0.1445614 ,  0.12350877],
       [ 0.14142259,  0.06694561,  0.07698745,  0.11129707,  0.10209205, 0.13472803,  0.11464435,  0.13974895,  0.11213389]])

for k = 16

for k = 20
"""
# Run HMM with given k components and covariance type
# Also saves a pickle for future use
def runHMM(k, cov_type='diag'):
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')

	print "Starting HMM"
	start_time = time.time()
	model = hmm.GaussianHMM(n_components=k, covariance_type=cov_type)
	X = observations.flatten().reshape(-1, 1)
	lengths = [1000] * 10000
	model.fit(X, lengths)
	print "Done running HMM"

	joblib.dump(model, "hmm%i_%s.pkl" % (k, cov_type))
	print("--- %s seconds ---" % (time.time() - start_time))

# Get predictions from model
def getPredictedStates(model):
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	X = observations.flatten().reshape(-1, 1)
	predictedStates = model.predict(X)

	return np.reshape(predictedStates, (10000,1000))

def linearRegression():
	sortedLabels = np.genfromtxt("../LabelSorted.csv", delimiter=',')
	regr = linear_model.LinearRegression()
	x_train = []
	y_train = []
	x_train.append(993)
	x_train.append(999)
	x_train.append(1000)
	y_train.append((sortedLabels[297, 2], sortedLabels[297, 3]))
	y_train.append((sortedLabels[298,2], sortedLabels[298, 3]))
	y_train.append((sortedLabels[299, 2], sortedLabels[299, 3]))

	regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1,1))

	y_pred = regr.predict(1001)
	print y_pred


def run():
	return
	# observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	# writeLabelSortedCSV()

	# print "starting hmm"
	# start_time = time.time()
	# k = 20
	# model = hmm.GaussianHMM(n_components=k)
	# model.fit(observations)
	# print "done running hmm"
	# joblib.dump(model, "hmm"+str(k)+".pkl")
	# print("--- %s seconds ---" % (time.time() - start_time))
	# model = joblib.load("hmm20.pkl")
	# print model.predict(observations)
	# sortedLabels = np.genfromtxt("../LabelSorted.csv", delimiter=',')
	# x = np.array([993, 999, 1000])
	# y1 = np.array([sortedLabels[297, 2], sortedLabels[298, 2], sortedLabels[299,2]])
	# y2 = np.array([sortedLabels[297, 3], sortedLabels[298, 3], sortedLabels[299,3]])
	# A = np.vstack([x, np.ones(len(x))]).T

	# m1, c1 = np.linalg.lstsq(A, y1)[0]
	# m2, c2 = np.linalg.lstsq(A, y2)[0]

	# x = m1 * 1001 + c1
	# y = m2 * 1001 + c1
	# print x, y
	# # linearRegression()

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)

	# predDict = {} # Prediction -> [Run # ...]

	# for idx, pred in enumerate(predictions):
	# 	run = idx + 1
	# 	predDict.setdefault(pred, []).append(run)


	# labelsDict = loadDict('labels_dict.pkl')
	model = joblib.load('hmm10_diag.pkl')
	last4000last5 = np.zeros((4000, 5))
	predictedStates = getPredictedStates(model)
	last4000last5 = predictedStates[6000:,-5:]
	print last4000last5
	# plotPredictedStates(predictedStates)


	# plotStates(predictions)

	# d = {}
	# predRunsDict = {}
	# for idx, pred in enumerate(predictions[:6000]):
	# 	run = idx + 1
	# 	steps = labelsDict[run]
	# 	d.setdefault(pred, []).append(steps)
	# 	predRunsDict.setdefault(pred, []).append(run)




	# # finalLabels = []
	# predDict = {}

	# for run, vals in labelsDict.items():
	# 	step, (x, y), angle = vals[-1]
	# 	if step == 1000:
	# 		pred = predictions[run-1]
	# 		# finalLabels.append((run, step, (x, y), angle))
	# 		predDict.setdefault(pred, []).append((run, step, (x, y), angle))


	# coords = []
	# for _, vals in labelsDict.items():
	# 	newVals = [(x, y) for step, (x, y), angle in vals if 200 <= step <= 205]
	# 	coords.extend(newVals)

	# xs = [x for x, _ in coords]
	# ys = [y for _, y in coords]
	# plt.plot(xs, ys, 'ro')
	# plt.show()








	


