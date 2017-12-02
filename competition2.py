import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm	
from math import *
from sklearn.externals import joblib
import pickle
import operator
from scipy.optimize import curve_fit
from numpy import pi, r_
from scipy import optimize
from scipy.optimize import leastsq
import numpy, scipy.optimize
from sklearn import linear_model
from shapely.geometry import LineString
from shapely.geometry import Point

"""
First column of observations matrix
- Angle after first step

Average angle: 0.83614049900000009
Min angle: 0.75205999999999995
Max angle: 0.93271000000000004

-- All observations --
Average angle: 0.78153661932199991
"""

"""
- Find peak points and valley points
"""

observations = np.genfromtxt("../Observations.csv", delimiter = ',')
labels = np.genfromtxt("../Label.csv", delimiter=',')
sortedLabels = np.genfromtxt("../LabelSorted.csv", delimiter=',')

############
# PLOTTING #
############

def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.show()

# Plot angles at all steps for run i (starting at 1)
def plotAnglesAtRun(run):
	angles = observations[run-1]
	avgAngle = np.mean(angles)

	# Plot actual observation angles
	xs = range(1, 1001)
	plt.plot(xs, angles)

	# Plot labelled points for run
	labelsDict = createLabelsDict()

	if run in labelsDict:
		steps = labelsDict[run]
		for step, _, angle in steps:
			plt.scatter(step, angle, color='red', marker='.')

	# Plot midline
	plt.plot([1, 1001], [avgAngle, avgAngle], linestyle='solid')

	# Plot division lines
	for i in range(0, angles.shape[0], 50):
		plt.plot([i, i], [0, 1.4], linestyle='solid')

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

# Plot the bot movement based on labelled data
def plotBotMovement():

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

# Plot the distribution of observed angles
# Min angle = 0.12031 ~ 6.893255233 degrees
# Max angle = 1.4424 ~ 82.6434324 degrees
def plotObservedAngleDistribution():
	OM = createObservationMatrix()
	angles = OM.flatten().reshape(-1, 1)
	plt.hist(angles, normed=False, bins=50)
	plt.show()


# Plot the states that the labels are assigned to
# Currently supports up to HMM 25
def plotPredictedStates(predictedStates, centroidMapping=None, numStates=10):
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
		if 1000 <= run < 2000:
			run = int(run) - 1000
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
	plt.plot([0, 3], [0, 3*tan(0.4039724)])
	plt.plot([0, 3], [0, 3*tan(0.6232251)])
	plt.plot([0, 2], [0, 2*tan(1.01099311)])
	plt.plot([0, 1], [0, 1*tan(1.20409)])
	plt.show()


#Plots the unit circle with centers (x1, y1) and (x2, y2)
def plotUnitCircle(x1, y1, x2, y2):
	# fig = plt.figure()
	# ax = fig.add_subplot(1, 1, 1)
	# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
	# ax.add_patch(circ)
	plt.scatter(x1, y1, color='red', marker='.')
	plt.scatter(x2, y2, color='blue', marker='.')
	plt.show()

def plotXYAngle(labelsDict, run, flipY=False):
	labels = labelsDict[run]
	avgAngle = 0.78153661932199991 # Average angle of all observations

	xs, ys, angles = [], [], []
	for _, (x, y), angle in labels:
		xs.append(x)
		ys.append(y)
		angles.append(angle)

	for angle, x in zip(angles, xs):
		plt.plot(angle, x, color='red', marker='.')

	if flipY:
		angleDiff = [a - avgAngle for a in angles]
		angles = [avgAngle - diff for diff in angleDiff]

	for angle, y in zip(angles, ys):
		plt.plot(angle, y, color='blue', marker='.')

	plotMidLine()

	plt.xlabel('Angle')
	plt.ylabel('X = red, Y = blue')

def plotMidLine():
	plt.plot([avgAngle, avgAngle], [0, 3], linestyle='solid')


#################
# PREPROCESSING #
#################

# Create 6000 x 1000 location matrix from Labels.csv
# M[i][j] = (x, y) --> At Run i + 1 and Step j + 1, bot is a location (x, y)
def createLocationMatrix():
	M = np.zeros((6000, 1000), dtype = 'float32, float32')

	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		M[int(run-1), int(step-1)] = (x+1.5, y+1.5)

	return M

# Create the labels dictionary
# Dict format: {Run: [(Step1, (x1, y1)) ... ]}
def createLabelsDict():
	locations = {}
	for i in range(labels.shape[0]):
		run, step, x, y = labels[i, :]
		run, step = int(run), int(step)
		x, y = x + 1.5, y + 1.5 # Add alpha/beta offsets
		observation = observations[run-1, step-1]
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
def createSubmission(predLocations, fileName):
	print 'creating submission file...'
	with open(fileName, 'wb') as f:
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

# Write to CSV the 10000 x 1000 predicted states matrix
def writeTruncPredictedStatesCSV(k=10):
	model = joblib.load('trunc_hmm%i_diag.pkl' % k)
	predictedStates = getTruncPredictedStates(model)
	np.savetxt('trunc_hmm%i_predicted_states.csv' % k, predictedStates, fmt='%i', delimiter=",")

# Load 10000 x 1000 predicted states matrix from CSV
def loadTruncPredictedStatesCSV(k=10):
	predictedStates = np.genfromtxt("trunc_hmm%i_predicted_states.csv" % k, delimiter=',')

	return predictedStates

################
# CALCULATIONS #
################
#Calculate the 1001th angle using linear regression on last 3 angles
def calculate1001AngleLinearRegression():
	regr = linear_model.LinearRegression()
	x_train = [998, 999, 1000]
	angle1001predictions = []

	for i in range(observations.shape[0]):
		y_train = []
		for j in range(3,0,-1):
			y_train.append(observations[i, -j])
		regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1,1))
		angle1001predictions.append(round(regr.predict(1001)[0][0], 8))

	for idx, i in enumerate(angle1001predictions):
		if i < 0.12031:
			diff = 0.12031 - i
			angle1001predictions[idx] = 0.12031 + diff
		elif i > 1.4424:
			diff = i - 1.4424
			angle1001predictions[idx] = 1.4424 - diff

	return angle1001predictions

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
	avgAngleDiffs = []
	minAngleDiff = float('inf')
	maxAngleDiff = float('-inf')
	totalAngleDiffs = []
	for row in observations:
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

# Return a RMSE score given 10000 x 1000 predictions
def evaluate(predictions):
	rmse = 0.0
	for label in sortedLabels:
		run, step, x, y = label
		run, step = int(run), int(step)
		predX, predY = predictions[run-1, step-1]
		rmseX = sqrt(mean_squared_error(x, predX))
		rmseY = sqrt(mean_squared_error(y, predY))
		rmse += rmseX + rmseY

	return rmse

def findIntersectionLineCircle(angle):
	A = tan(angle)**2 + 1
	B = 2 * (-tan(angle) * 1.5 - 1.5)
	C = 1.5**2 - 1 + 1.5**2

	if B**2 - 4*A*C < 0:
		return 0,0
	else:
		x_intersect1 = (-B + sqrt(B**2 - 4*A*C))/(2*A)
		x_intersect2 = (-B - sqrt(B**2 - 4*A*C))/(2*A)
		y_intersect1 = tan(angle) * x_intersect1
		y_intersect2 = tan(angle) * x_intersect2

		return (x_intersect1, y_intersect1), (x_intersect2, y_intersect2)

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

# Get direction of movement: -1 for decreasing angle, +1 for increasing angle
def getDirection(angleSeq):
	direction = None
	for i in range(len(angleSeq) - 1, 0, -1):
		if direction: continue

		prev, curr = angleSeq[i-1], angleSeq[i]
		if prev < curr:
			direction = 1
		elif prev > curr:
			direction = -1

	if not direction:
		direction = 1

	return direction

# Get most likely direction of 1001th point
def getFinalDirections():
	directions = []
	for row in observations:
		lastAngleSeq = row[-5:]
		direction = getDirection(lastAngleSeq)
		directions.append(direction)

	return directions

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

# Get predictions from model
def getPredictedStates(model):
	X = observations.flatten().reshape(-1, 1)
	lengths = [1000] * 10000
	predictedStates = model.predict(X, lengths)

	return np.reshape(predictedStates, (10000,1000))

def getTruncPredictedStates(model):
	truncObservations = observations[1000:2000,:]
	X = truncObservations.flatten().reshape(-1, 1)
	lengths = [1000] * 1000
	predictedStates = model.predict(X, lengths)

	return np.reshape(predictedStates, (1000,1000))

# Get the two centroids for each state
def getStateCentroids(predictedStates, mapping, numStates=10):
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

def performLinearRegression(x_train, y_train, x_test):
	regr = linear_model.LinearRegression()

	regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1,1))
	return regr.predict(x_test)

# Given the angle and direction of movement, predict the x, y location
def predictLocation(angle, direction, anglesMapping):
	angles = sorted(list(anglesMapping.keys()))

	angleBuffer = 0.025
	angleRange = (angle - angleBuffer, angle + angleBuffer)
	points = []
	for a in angles:
		if a >= angleRange[0] and a <= angleRange[1]:
			section = 'top' if direction == 1 else 'bot'
			points += anglesMapping[a][section]

	x_c = np.mean(np.array([x for x, _ in points]))
	y_c = np.mean(np.array([y for _, y in points]))

	# # Project onto line y = tan(angle) * x
	# c = y_c + x_c * tan(angle)
	# x_p = c / (tan(angle) + 1 / tan(angle))
	# y_p = tan(angle) * x_p


	return (x_c, y_c)

##############
# ALGORITHMS #
##############

#Approximate (x, y) locations using the unit circle and the 1000th angle
def approximateUnitCircle():
	lastObservation = observations[6000:, -1]
	locations = []
	directions = getFinalDirections()
	directions = directions[6000:]

	for idx, obs in enumerate(lastObservation):
		if directions[idx] == 1:
			obs = obs + 0.05
		else:
			obs = obs - 0.05

		one, two = findIntersectionLineCircle(obs)

		if one == 0 and two == 0:
			if obs > 1.2762808455:
				locations.append((0.543057, 1.790276))
			elif obs < 0.29451544361:
				locations.append((1.790276, 0.543057))
		else:
			x_one, y_one = one
			x_two, y_two = two
			if  y_one > 2.333333 - x_one and directions[idx] == 1:
				locations.append((x_one, y_one))
			elif y_one <= 2.333333 - x_one and directions[idx] == -1:
				locations.append((x_one, y_one))
			elif y_two > 2.3333333 - x_two and directions[idx] == 1: 
				locations.append((x_two, y_two))
			elif y_two <= 2.3333333 - x_two and directions[idx] == -1: 
				locations.append((x_two, y_two))

	return locations

# Run HMM with given k components and covariance type
# Also saves a pickle for future use
def runHMM(k, cov_type='diag'):
	print "Starting HMM"
	start_time = time.time()
	model = hmm.GaussianHMM(n_components=k, covariance_type=cov_type)
	X = observations.flatten().reshape(-1, 1)
	lengths = [1000] * 10000
	model.fit(X, lengths)
	print "Done running HMM"

	joblib.dump(model, "hmm%i_%s.pkl" % (k, cov_type))
	print("--- %s seconds ---" % (time.time() - start_time))

# Run HMM with given k components and covariance type
# Also saves a pickle for future use
def runTruncatedHMM(k, cov_type='diag'):
	truncObservations = observations[:1000,:]

	# np.savetxt('newObs,csv', newObs, fmt='%f', delimiter=",")
	print "Starting HMM"
	start_time = time.time()
	model = hmm.GaussianHMM(n_components=k, covariance_type=cov_type)
	X = truncObservations.flatten().reshape(-1, 1)
	lengths = [1000] * 1000
	model.fit(X, lengths)
	print "Done running HMM"

	joblib.dump(model, "trunc_hmm%i_%s_run1000.pkl" % (k, cov_type))
	print("--- %s seconds ---" % (time.time() - start_time))

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

	createSubmission(predLocations, './hmm10_submission.csv')

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

	createSubmission(predLocations, './hmm16_submission.csv')

def newHmm16():
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

	last4000Direction = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	last4000Angles = observations[6000:,-1]

	anglesMapping = loadDict('./angle_mapping.csv')
	angles = sorted(list(anglesMapping.keys()))

	predLocations = []
	angleDelta = 0.05
	for angle, direction in zip(last4000Angles, last4000Direction):
		angle += angleDelta * direction
		predLoc = predictLocation(angle, direction, anglesMapping)
		predLocations.append(predLoc)

	createSubmission(predLocations, './hmm16_submission_dup.csv')

def unitHmm16():
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

	last4000Direction = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)

	print 'getting predicted angles...'
	# predicted4000Angles = getPredictedAngles2()
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	predicted4000Angles = observations[6000:,-1]

	print 'predicting locations...'
	predLocations = []
	angleDelta = 0.05
	for angle, direction in zip(predicted4000Angles, last4000Direction):
		angle += angleDelta * direction
		predLoc = predLocOnCircle(angle, direction)
		predLocations.append(predLoc)

	createSubmission(predLocations, './hmm16_submission_unit_3.csv')


def betterHmm16():
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
	last4000Direction = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)

	print 'getting predicted angles...'
	predicted4000Angles = getPredictedAngles2()
	anglesMapping = loadDict('./angle_mapping.csv')

	print 'predicting locations...'
	predLocations = []
	for angle, direction in zip(predicted4000Angles, last4000Direction):
		predLoc = predictLocation(angle, direction, anglesMapping)
		predLocations.append(predLoc)

	createSubmission(predLocations, './hmm16_submission_pred_angles_3.csv')

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
	t = np.linspace(1, 1000, 1000, dtype = 'int32')

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

# Get a 10000 x 1 array of the 1001th predicted angles
def getPredictedAngles():
	predAngles = []
	for angles in observations:
		peaks, valleys = findPeaksValleys(angles)
		amplitude, period, hShift, vShift = getGuessParameters(peaks, valleys)
		fit_amplitude, fit_period, fit_hShift, fit_vShift = getFittedParameters(angles, amplitude, period, hShift, vShift)
		predAngle = fit_amplitude * np.sin(fit_period * (1001 + fit_hShift)) + fit_vShift
		predAngles.append(predAngle)

	return predAngles

# Get 1001th predicted angles
def getPredictedAngles2():
	lastFour = observations[:, -4:]
	angleBuckets = [0.4039724, 0.6232251, 1.01099311, 1.20409]

	predAngles = []
	for run in lastFour:
		predAngle = 0
		if run[3] < angleBuckets[0] or run[3] > angleBuckets[3]:
			# Avg angle
			predAngle = np.mean(run[1:])
		elif run[3] < angleBuckets[1] or run[3] > angleBuckets[2]:
			# Avg delta
			diff = run[3]-run[2] + run[2]-run[1] + run[1]-run[0]
			diff /= 3
			predAngle = run[3] + diff
		else:
			# Middle zone, keep delta trend
			prevDelta = run[3]-run[2]
			predAngle = run[3] + prevDelta

		predAngles.append(predAngle)

	return np.array(predAngles)

def performLinearRegression(x_train, y_train, x_test):
	regr = linear_model.LinearRegression()
	regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1,1))
	return regr.coef_, regr.intercept_, regr.predict(np.array(x_test).reshape(-1, 1))

def predictedAnglesLinearRegression():
	angle_predictions = getPredictedAngles()	
	LM = createLocationMatrix()
	x_train_bot = []
	y_train_bot = []
	angle_train_bot = []
	x_train_top = []
	y_train_top = []
	angle_train_top = []
	for i in range(LM.shape[0]):
		counterTop = 3
		counterBot = 3
		for j in range(LM.shape[1]-1, -1, -1):
			x, y = LM[i, j]
			if counterTop != 0:
				if x == 0.0 and y == 0.0:
					continue
				else:
					#top half
					if y > 2.5 - x:
						x_train_top.append(x)
						y_train_top.append(y)
						angle_train_top.append(observations[i,j])
						counterTop -= 1

			if counterBot != 0:
				if x == 0.0 and y == 0.0:
					continue
				else:
					#bot half
					if y <= 2.5 - x:
						x_train_bot.append(x)
						y_train_bot.append(y)
						angle_train_bot.append(observations[i,j])
						counterBot -= 1

			if counterTop == 0 and counterBot == 0:
				break
	print "linear regression"
	m_bot_x, c_bot_x, predicted_bot_x = performLinearRegression(angle_train_bot, x_train_bot, angle_predictions[6000:])
	m_bot_y, c_bot_y, predicted_bot_y = performLinearRegression(angle_train_bot, y_train_bot, angle_predictions[6000:])
	m_top_x, c_top_x, predicted_top_x = performLinearRegression(angle_train_top, x_train_top, angle_predictions[6000:])
	m_top_y, c_top_y, predicted_top_y = performLinearRegression(angle_train_top, y_train_top, angle_predictions[6000:])

	predictedStates = loadPredictedStatesCSV(16)
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

	print "directions"
	directions = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)
	predLocations = []
	for idx, d in enumerate(directions):
		if d == 1:
			predLocations.append((predicted_top_x[idx][0], predicted_top_y[idx][0]))
		else:
			predLocations.append((predicted_bot_x[idx][0], predicted_bot_y[idx][0]))

	print "creating submission"
	createSubmission(predLocations,"hmm16_submission_regression.csv")
	# plt.plot(angle_train_bot, x_train_bot, 'ro')
	# plt.plot([0,2], [c_bot_x, m_bot_x*2+c_bot_x])
	# plt.plot(angle_train_bot, y_train_bot, 'bo')
	# plt.plot([0,2], [c_bot_y, m_bot_y*2+c_bot_y])
	# plt.plot(angle_train_top, x_train_top, 'go')
	# plt.plot([0,2], [c_top_x, m_top_x*2+c_top_x])
	# plt.plot(angle_train_top, y_train_top, 'yo')
	# plt.plot([0,2], [c_top_y, m_top_y*2+c_top_y])
	# plt.show()

	# centroidMapping = loadDict('hmm16_centroid_mapping.csv')
	# final_locations = circleLineCalculation(angle_predictions, centroidMapping)

# Find closest angle to given angle
def findClosestAngle(angle):
	anglesDict = loadDict('angles_dict.pkl')

	if angle in anglesDict: return angle

	# Find closest angle
	angles = anglesDict.keys()

	closestAngle = angles[0]
	minDiff = abs(angles[0] - angle)
	for a in angles[1:]:
		if abs(angle - a) < minDiff:
			minDiff = abs(angle - a)
			closestAngle = a

	return a

# Write angles dictionary
# angle -> {direction -> [locations...]}
def writeAnglesDict():
	anglesDict = {}
	for label in labels:
		run, step, x, y = label
		angle = observations[int(run)-1, int(step)-1]
		angleSeq = observations[int(run)-1, int(step)-6:int(step)-1]
		direction = getDirection(angleSeq)
		anglesDict.setdefault(angle, {}).setdefault(direction, []).append((x, y))

	writeDict(anglesDict, 'angles_dict.pkl')

# Returns dictionary with key as angle and value as {bot: [], top: []}, holding
# arrays of labeled locations on that angle
def mapAnglesToLocations():
	angleMapping = {}
	for label in labels:
		run, step, x, y = label
		x, y = x + 1.5, y + 1.5
		angle = observations[int(run)-1, int(step)-1]
		section =  'top' if (y > 2.5 - x) else 'bot'
		angleMapping.setdefault(angle, {'bot': [], 'top': []})[section].append((x,y))

	# writeDict(angleMapping, './angle_mapping.csv')
	return angleMapping

def dist(p1, p2):
	x_d = mean_squared_error([p1[0]], [p2[0]])
	y_d = mean_squared_error([p2[1]], [p2[1]])
	return sqrt(x_d + y_d)

def compareSubmissions(filename1, filename2):
	with open(filename1, 'r') as f1:
		with open(filename2, 'r') as f2:
			lines1 = f1.readlines()
			lines2 = f2.readlines()

			totalDistance = 0
			for i in range(1, len(lines1), 2):
				x1 = float(lines1[i].split(',')[1])
				y1 = float(lines1[i+1].split(',')[1])
				x2 = float(lines2[i].split(',')[1])
				y2 = float(lines2[i+1].split(',')[1])

				totalDistance += dist((x1, y1), (x2, y2))
			totalDistance = sqrt(totalDistance / (len(lines1)-1))

			print 'Total distance', totalDistance

def compare1001AngleTo1000Angle():
	# predAngles = linearRegression()
	predAngles = getPredictedAngles2()
	OM = createObservationMatrix()[:, -3:]

	with open('./angle_comparison.csv', 'wb') as f:
		for idx, angle in enumerate(predAngles):
			array = list(OM[idx])
			array.append(angle)

			f.write(','.join(map(str, array)))
			f.write('\n')

# Find center point from labelled data
def findCenterPoint():
	xs, ys = [], []
	for label in labels:
		run, step, x, y = label
		xs.append(x+1.5)
		ys.append(y+1.5)

	avgX = np.mean(xs)
	avgY = np.mean(ys)

	return (avgX, avgY)

# Get ellipse equation for donut
def getEllipseEquation(x):
	centerX = 1.50203568876
	centerY = 1.49502535473

	minY = 0.50338353166532002
	minX = 0.45720515002048062

	a = centerX - minX # 1.0448305387395194
	b = centerY - minY # 0.9916418230646801

	y1 = sqrt((b ** 2) * (1 - (((x - centerX) ** 2) / (a ** 2)))) + centerY
	y2 = (-1 * sqrt((b ** 2) * (1 - (((x - centerX) ** 2) / (a ** 2))))) + centerY

	return (y1, y2)

def hmm16Annie():
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

	directions = getLast4000Direction(predictedStates, hmm16_pred_actual_mapping)

	print 'getting predicted angles...'
	# predicted4000Angles = getPredictedAngles2()
	predAngles = observations[6000:,-1]

	# predAngles = loadDict('predicted_angles2.pkl')
	# directions = loadDict('final_directions.pkl')

	predLocations = []
	for angle, direction in zip(predAngles, directions):
		intersectPts = findCircleIntersection(angle)

		if intersectPts:
			if len(intersectPts) == 1:
				x, y = intersectPts[0]
				predLocations.append((x, y))
			elif len(intersectPts) == 2:
				(x1, y1), (x2, y2) = intersectPts
				x, y = (x2, y2) if direction == 1 else (x1, y1)
				predLocations.append((x, y))
			else:
				raise Exception('Too many intersection points')
		else:
			upperAngle = 1.2762808455
			lowerAngle = 0.29451544361

			if angle > upperAngle:
				x, y = (0.543057, 1.790276)
				predLocations.append((x, y))
			else:
				x, y = (1.790276, 0.543057)
				predLocations.append((x, y))

	createSubmission(predLocations, './hmm16_submission_annie.csv')

"""
no intersection: 1161
one intersection: 0
two intersections: 8839
"""
def run():
	# predAngles = loadDict('predicted_angles2.pkl')
	predAngles = list(observations[:, -1])
	directions = loadDict('final_directions.pkl')

	predLocations = []
	for angle, direction in zip(predAngles, directions):
		intersectPts = findCircleIntersection(angle)

		if intersectPts:
			if len(intersectPts) == 1:
				x, y = intersectPts[0]
				predLocations.append((x, y))
			elif len(intersectPts) == 2:
				(x1, y1), (x2, y2) = intersectPts
				x, y = (x2, y2) if direction == 1 else (x1, y1)
				predLocations.append((x, y))
			else:
				raise Exception('Too many intersection points')
		else:
			upperAngle = 1.2762808455
			lowerAngle = 0.29451544361

			if angle > upperAngle:
				x, y = (0.543057, 1.790276)
				predLocations.append((x, y))
			else:
				x, y = (1.790276, 0.543057)
				predLocations.append((x, y))

	createSubmission(predLocations[6000:], 'pred_angles_unit_circle.csv')


	# writeDict(predLocations, 'final_pred_locations.pkl')

	# labels = np.genfromtxt("../LabelSorted.csv", delimiter=',')
	# predLocations = loadDict('final_pred_locations.pkl')

	# for location in predLocations:
	# 	if not location: continue
	# 	x, y = location
	# 	plt.scatter(x, y, marker='.')

	# labelsDict = {}
	# for label in labels:
	# 	run, step, x, y = label
	# 	if int(step) == 1000:
	# 		labelsDict[int(run)] = (x + 1.5, y + 1.5)

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	plotBotMovement()
	plotUnitCircle(1.5, 1.5, 1.5, 1.5)

	# predictedAngles = getPredictedAngles()
	# directions = getFinalDirections()
	# predictedAngles = loadDict('predicted_angles.pkl')
	# anglesMapping = loadDict('angles_dict.pkl')
	# compareSubmissions('./hmm16_submission_2.csv', './hmm16_submission_annie.csv')
	# newHmm16()
	# compare1001AngleTo1000Angle()
	# unitHmm16()
	# hmm16Annie()





































