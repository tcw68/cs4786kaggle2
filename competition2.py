import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import Markov
from math import *
from sklearn.externals import joblib
import time
import pickle
from random import randint
import operator
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from numpy import pi, r_
from scipy import optimize
from scipy.optimize import leastsq
import numpy, scipy.optimize
from sklearn import linear_model

"""
First column of observations matrix
- Angle after first step

Average angle: 0.83614049900000009
Min angle: 0.75205999999999995
Max angle: 0.93271000000000004
"""

"""
- Find peak points and valley points
"""

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
# Currently supports up to HMM 25
def plotPredictedStates(predictedStates, centroidMapping=None, numStates=10):
	labels = np.genfromtxt("../Label.csv", delimiter=',')

	# Map each state to a distinct RGB color
	cmap = ['#ff1111', '#ff8b11', '#fff311', '#9bff11', '#11ff88', 
			'#11f7ff', '#1160ff', '#7011ff', '#ff11e7', '#ff114c', 
			'#000000', '#723416', '#0bc68b', '#9003af', '#a5a4a5', 
			'#1a87ba', '#a64c79', '#8a9fc6', '#d0e596', '#036f4b', 
			'#fe98b2', '#ccbaa9', '#708965', '#47574d', '#ffc300']

	cmap = cmap[:numStates]

	xVals = [[] for _ in range(numStates)]
	yVals = [[] for _ in range(numStates)]
	for label in labels:
		run, step, x, y = label
		nextState = int(predictedStates[int(run) - 1, int(step) - 1])
		xVals[nextState].append(x + 1.5)
		yVals[nextState].append(y + 1.5)

	for i in range(numStates):
		plt.scatter(xVals[i], yVals[i], color=cmap[i], marker='.')

	if centroidMapping:
		xCentroids, yCentroids = [], []
		for botCoord, topCoord in centroidMapping.values():
			botX, botY = botCoord
			topX, topY = topCoord
			xCentroids.extend([botX, topX])
			yCentroids.extend([botY, topY])

		plt.plot(xCentroids, yCentroids, 'wo')

	plt.plot([0, 2.5], [2.5, 0], linestyle='solid') 
	plt.show()

# Plot the distribution of observed angles
# Min angle = 0.12031 ~ 6.893255233 degrees
# Max angle = 1.4424 ~ 82.6434324 degrees
def plotObservedAngleDistribution():
	OM = createObservationMatrix()
	angles = OM.flatten().reshape(-1, 1)
	plt.hist(angles, normed=False, bins=50)
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
 
# Create submission file using 4000 (x, y) predicted location points
def createSubmission(predLocations, k):
	with open('hmm%i_submission.csv' % k, 'wb') as f:
		f.write('Id,Value\n')

		for i, (x, y) in enumerate(predLocations):
			xLine = ','.join([str(i+6001)+'x', str(x-1.5)])
			yLine = ','.join([str(i+6001)+'y', str(y-1.5)])
			f.write(xLine + '\n')
			f.write(yLine + '\n')
			

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

# Write to CSV the 10000 x 1000 predicted states matrix
def writePredictedStatesCSV(k=10):
	model = joblib.load('hmm%i_diag.pkl' % k)
	predictedStates = getPredictedStates(model)
	np.savetxt('hmm%i_predicted_states.csv' % k, predictedStates, fmt='%i', delimiter=",")

# Load 10000 x 1000 predicted states matrix from CSV
def loadPredictedStatesCSV(k=10):
	predictedStates = np.genfromtxt("hmm%i_predicted_states.csv" % k, delimiter=',')

	return predictedStates

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
# Min step size = 0.00150479234448
# Max step size = 0.613946069316
def calculateAverageStepSize():
	sortedLabels = np.genfromtxt("../LabelSorted.csv", delimiter=',')
	distances = []

	for i in range(1, sortedLabels.shape[0]):
		if sortedLabels[i-1, 0] == sortedLabels[i, 0]:
			if sortedLabels[i-1, 1] + 1 == sortedLabels[i, 1]:
				distances.append(np.linalg.norm(sortedLabels[i-1,2:]-sortedLabels[i,2:]))

	distances = np.array(distances)

	# Plot bar graph
	# plt.hist(distances, normed=True, bins=20)
	# plt.show()

	return np.mean(distances)

# Calculate the average angle difference between consecutive angles
# Avg angle diff = 0.050048676122122118
# Min angle diff = 0.0
# Max angle diff = 0.4062
def calculateAverageAngleDiff():
	OM = createObservationMatrix()

	avgAngleDiffs = []
	minAngleDiff = float('inf')
	maxAngleDiff = float('-inf')
	totalAngleDiffs = []
	for row in OM:
		angleDiffs = []
		for i in range(1, row.shape[0]):
			angleDiff = abs(row[i] - row[i-1])
			angleDiffs.append(angleDiff)

			minAngleDiff = min(minAngleDiff, angleDiff)
			maxAngleDiff = max(maxAngleDiff, angleDiff)
			totalAngleDiffs.append(angleDiff)

		angleDiffs = np.array(angleDiffs)
		avgAngleDiffs.append(np.mean(angleDiffs))

	avgAngleDiffs = np.array(avgAngleDiffs)

	print "Min angle diff: ", minAngleDiff
	print "Max angle diff: ", maxAngleDiff
	print "Total angle diff: ", totalAngleDiffs

	# Plot bar graph
	# plt.hist(avgAngleDiffs, normed=True, bins=20)
	# plt.show()

	return np.mean(avgAngleDiffs)

# Get the two centroids for each state 
def getStateCentroids(predictedStates, mapping, numStates=10):
	labels = np.genfromtxt("../Label.csv", delimiter=',')

	topStateCoords = [[] for _ in range(numStates)]
	botStateCoords = [[] for _ in range(numStates)]
	for label in labels:
		run, step, x, y = label
		nextState = predictedStates[int(run) - 1, int(step) - 1]
		mappedState = mapping[nextState]
		x, y = x + 1.5, y + 1.5

		# Division line: y = 2.5 - x
		if y > 2.5 - x: # Above line
			topStateCoords[mappedState].append((x, y))
		else: # Below line
			botStateCoords[mappedState].append((x, y))

	centroidMapping = {}

	for i in range(numStates):
		topCoords = topStateCoords[i]
		topX = np.mean(np.array([x for x, _ in topCoords]))
		topY = np.mean(np.array([y for _, y in topCoords]))

		botCoords = botStateCoords[i]
		botX = np.mean(np.array([x for x, _ in botCoords]))
		botY = np.mean(np.array([y for _, y in botCoords]))

		centroidMapping[i] = [(botX, botY), (topX, topY)]

	return centroidMapping

##############
# ALGORITHMS #
##############

# Run HMM with given k components and covariance type
# Also saves a pickle for future use
def runHMM(k, cov_type='diag'):
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	newObs = np.zeros((10000,1000,4))

	for i in range(observations.shape[0]):
		for j in range(observations.shape[1]):
			if j < 3:
				newObs[i, j, :] = observations[i, j]
			else:
				newObs[i, j, 0] = observations[i, j]
				newObs[i, j, 1] = observations[i, j-1]
				newObs[i, j, 2] = observations[i, j-2]
				newObs[i, j, 3] = observations[i, j-3]

	# np.savetxt('newObs,csv', newObs, fmt='%f', delimiter=",")
	print "Starting HMM"
	start_time = time.time()
	model = hmm.GaussianHMM(n_components=k, covariance_type=cov_type)
	X = newObs.flatten().reshape(-1, 1)
	lengths = [1000] * 10000
	model.fit(X, lengths)
	print "Done running HMM"

	joblib.dump(model, "hmm%i_%s_4angles.pkl" % (k, cov_type))
	print("--- %s seconds ---" % (time.time() - start_time))

# Get predictions from model
def getPredictedStates(model):
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	X = observations.flatten().reshape(-1, 1)
	lengths = [1000] * 10000
	predictedStates = model.predict(X, lengths)

	return np.reshape(predictedStates, (10000,1000))

def getPredictedStatesNewObs(model):
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	X = observations.flatten().reshape(-1, 1)
	lengths = [[4]*1000] * 10000
	predictedStates = model.predict(X, lengths)

	return np.reshape(predictedStates, (10000,1000,4))
   
#Truncates/pads a float f to n decimal places without rounding

def linearRegression():
	observations = np.genfromtxt("../Observations.csv", delimiter=',')
	regr = linear_model.LinearRegression()
	x_train = [998, 999, 1000]
	angle1001predictions = []

	for i in range(observations.shape[0]):
		y_train = []
		for j in range(3,0,-1):
			y_train.append(observations[i, -j])
		regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1,1))
		angle1001predictions.append(round(regr.predict(1001)[0][0], 8))

	return angle1001predictions

def evaluate(y_actual, y_predicted):
	rms = sqrt(mean_squared_error(y_actual, y_predicted))
	return rms

def circleLineCalculation(angle_predictions):
	angle_predictions4000 = angle_predictions[6000:]

	for i in angle_predictions4000:
		 = math.tan(angle_predictions) * x

# Get whether the last 4000 points are increasing or decreasing states
def getLast4000Direction(predictedStates, mapping):
	last4000last5 = np.zeros((4000, 5))
	last4000last5 = predictedStates[6000:,-5:]

	for i in range(last4000last5.shape[0]):
		for j in range(last4000last5.shape[1]):
			last4000last5[i, j] = mapping[last4000last5[i, j]]

	directions = [] # -1 for decreasing, 1 for increasing
	for last5 in last4000last5:
		directionFound = False
		for i in range(len(last5) - 1, 0, -1):
			if directionFound: continue

			prev, curr = last5[i-1], last5[i]
			if prev < curr:
				directions.append(1)
				directionFound = True
			elif prev > curr:
				directions.append(-1)
				directionFound = True

		if not directionFound:
			directions.append(1)

	return directions

# Write HMM 10 submission file
def hmm10():
	hmm10_pred_actual_mapping = {
		0: 3,
		1: 7,
		2: 1,
		3: 9,
		4: 5,
		5: 4,
		6: 6,
		7: 2,
		8: 8,
		9: 0
	}

	predictedStates = loadPredictedStatesCSV()
	centroidMapping = loadDict('hmm10_centroid_mapping.pkl')

	last4000Direction = getLast4000Direction(predictedStates, hmm10_pred_actual_mapping)
	last4000States = predictedStates[6000:,-1]

	predLocations = []
	for state, direction in zip(last4000States, last4000Direction):
		botCoord, topCoord = centroidMapping[state]
		print topCoord
		if direction == 1:
			if state + 1 < 10:
				_, topCoordnext = centroidMapping[state+1]
				print topCoordnext
				xtopnext, ytopnext = topCoordnext
				print xtopnext, ytopnext
				xtop, ytop = topCoord
				print xtop, ytop

				d_prime = sqrt((xtopnext - xtop)**2 + (ytopnext - ytop)**2)

				predLocation = (xtop + (0.15 * (xtopnext - xtop) / d_prime), ytop + (0.15 * (ytopnext-ytop) / d_prime))

			elif state == 9:
				predLocation = botCoord
		elif direction == 0:
			if state - 1 > 0:
				botCoordnext, _ = centroidMapping[state-1]
				xbotnext, ybotnext = botCoordnext
				xbot, ybot = botCoord

				d_prime = sqrt((xbotnext - xbot)**2 + (ybotnext - ybot)**2)

				predLocation = (xbot + (0.15 * (xbotnext - xbot) / d_prime), ybot + (0.15 * (ybotnext-ybot) / d_prime))

			elif state == 0:
				predLocation = topCoord
		# predLocation = topCoord if direction == 1 else botCoord
		quit()
		predLocations.append(predLocation)

	createSubmission(predLocations)

# Write HMM 16 submission file
def hmm16():
	hmm16_pred_actual_mapping = {
		0: 1,
		1: 11,
		2: 6,
		3: 14,
		4: 9,
		5: 4,
		6: 13,
		7: 5,
		8: 10,
		9: 7,
		10: 0,
		11: 3,
		12: 12,
		13: 2,
		14: 15,
		15: 8
	}

	predictedStates = loadPredictedStatesCSV(16)
	centroidMapping = loadDict('hmm16_centroid_mapping.pkl')

	last4000Direction = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)
	last4000States = predictedStates[6000:,-1]

	predLocations = []
	for state, direction in zip(last4000States, last4000Direction):
		botCoord, topCoord = centroidMapping[state]
		predLocation = topCoord if direction == 1 else botCoord
		predLocations.append(predLocation)

	createSubmission(predLocations, 16)

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    # plt.show()

# labelledAngles is dictionary of key = angle and value = (run, step, (x, y))
# minMaxAngles is dictionary of key = angle and value = 4 coordinates
def getLabelledAngles():
	labelsDict = createLabelsDict()

	labelledAngles = {}
	for run, vals in labelsDict.items():
		for step, (x, y), angle in vals:
			labelledAngles.setdefault(angle, []).append((run, step, (x,y)))

	minMaxAngles = {}
	for angle, vals in labelledAngles.items():
		botMinX, botMinY = float('inf'), float('inf')
		botMaxX, botMaxY = float('-inf'), float('-inf')
		topMinX, topMinY = float('inf'), float('inf')
		topMaxX, topMaxY = float('-inf'), float('-inf')

		for run, step, (x, y) in vals:
			if y > 2.5 - x: # Above line
				if x <= topMinX and y <= topMinY:
					topMinX, topMinY = x, y
				if x >= topMaxX and y >= topMaxY:
					topMaxX, topMaxY = x, y
			else: # Below line
				if x <= botMinX and y <= botMinY:
					botMinX, botMinY = x, y
				if x >= botMaxX and y >= botMaxY:
					botMaxX, botMaxY = x, y

		minMaxAngles[angle] = [(botMinX, botMinY), (botMaxX, botMaxY), (topMinX, topMinY), (topMaxX, topMaxY)]

	return (labelledAngles, minMaxAngles)

# Find the peak and valley points given observation angles for a particular run
def findPeaksValleys(angles):
	peaks, valleys = [], []
	for i in range(0, angles.shape[0], 50):
		angleSection = angles[i:i+50]
		minIdx, minAngle = min(enumerate(angleSection), key=operator.itemgetter(1))
		maxIdx, maxAngle = max(enumerate(angleSection), key=operator.itemgetter(1))

		valleys.append((i + minIdx, minAngle))
		peaks.append((i + maxIdx, maxAngle))

	return (peaks, valleys)

# Given the angle, predict the x, y location
def predictLocation():
	pass

# Return a RMSE score given 10000 x 1000 predictions
def evaluate(predictions):
	labels = np.genfromtxt("../LabelSorted.csv", delimiter=',')

	rmse = 0.0
	for label in labels:
		run, step, x, y = label
		run, step = int(run), int(step)
		predX, predY = predictions[run-1, step-1]
		rmseX = sqrt(mean_squared_error(x, predX))
		rmseY = sqrt(mean_squared_error(y, predY))
		rmse += rmseX + rmseY

	return rmse

# Get the guess parameters for the sine function
def getGuessParameters(peaks, valleys):
	# Calculate amplitude
	peakAngles = [angle for _, angle in peaks]
	valleyAngles = [angle for _, angle in valleys]
	avgPeakAngle = np.mean(np.array(peakAngles))
	avgValleyAngle = np.mean(np.array(valleyAngles))
	amplitude = (avgPeakAngle - avgValleyAngle) / 2

	# Calculate period
	peakSteps = [steps for steps, _ in peaks]
	valleySteps = [steps for steps, _ in valleys]
	avgPeakStep = np.mean(np.array([y - x for x, y in zip(peakSteps, peakSteps[1:])]))
	avgValleyStep = np.mean(np.array([y - x for x, y in zip(valleySteps, valleySteps[1:])]))
	frequency = np.mean([avgPeakStep, avgValleyStep])
	period = (2 * pi) / frequency

	# Calculate horizontal and vertical shifts
	hShift = peakSteps[0] - (frequency / 2.0)
	vShift = (avgPeakAngle + avgValleyAngle) / 2.0

	return amplitude, period, hShift, vShift

# Get the fitted parameters
def getFittedParameters(data, guess_amplitude, guess_period, guess_hShift, guess_vShift):
	t = np.linspace(1, 1000, 1000, dtype='int32')

	# First estimate
	data_guess = guess_amplitude * np.sin(guess_period * (t + guess_hShift)) + guess_vShift

	# Define optimal function: minimize difference between actual data and guess
	opt_func = lambda x: guess_amplitude * np.sin(x[0] * (t + x[1])) + x[2] - data
	est_period, est_hShift, est_vShift = leastsq(opt_func, [guess_period, guess_hShift, guess_vShift])[0]
	est_amplitude = guess_amplitude

	# Fit curve using optimal parameters
	data_fit = est_amplitude * np.sin(est_period * (t + est_hShift)) + est_vShift

	# plt.plot(data, '.')
	# plt.plot(data_fit, label='after fitting')
	# plt.plot(data_guess, label='first guess')
	# plt.legend()
	# plt.show()

	return (est_amplitude, est_period, est_hShift, est_vShift)

# Plot angles at all steps for run i (starting at 1)
def plotAnglesAtRun(run):
	OM = createObservationMatrix()
	angles = OM[run-1]
	avgAngle = np.mean(angles)

	# Plot actual observation angles
	xs = range(1, 1001)
	plt.plot(xs, angles)

	# Plot labelled points for run
	labelsDict = createLabelsDict()
	steps = labelsDict[run]
	for step, _, angle in steps:
		plt.scatter(step, angle, color='red', marker='.')

	# Plot midline
	plt.plot([1, 1001], [avgAngle, avgAngle], linestyle='solid')

	# Plot division lines
	# for i in range(0, angles.shape[0], 50):
	# 	plt.plot([i, i], [0, 1.4], linestyle='solid')

	# Plot peaks and valleys
	peaks, valleys = findPeaksValleys(angles)

	for x, y in peaks:
		plt.scatter(x, y, color='green', marker='.')

	for x, y in valleys:
		plt.scatter(x, y, color='green', marker='.')

	# Plot fitted function
	amplitude, period, hShift, vShift = getGuessParameters(peaks, valleys)
	fit_amplitude, fit_period, fit_hShift, fit_vShift = getFittedParameters(angles, amplitude, period, hShift, vShift)
	fit_ys = [fit_amplitude * np.sin(fit_period * (x + fit_hShift)) + fit_vShift for x in xs]
	plt.plot(xs, fit_ys)

	plt.show()

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)

	OM = createObservationMatrix()
	angles = OM[run-1]

	peaks, valleys = findPeaksValleys(angles)
	amplitude, period, hShift, vShift = getGuessParameters(peaks, valleys)
	fit_amplitude, fit_period, fit_hShift, fit_vShift = getFittedParameters(angles, amplitude, period, hShift, vShift)
	





	











	






	
















	


