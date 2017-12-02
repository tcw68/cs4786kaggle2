from sklearn.externals import joblib
from sklearn.svm import SVR
import numpy as np

# Followed this example from sklearn library using RBF kernel:
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
def runSVR(sampleSize, numPrevAngles):
	labels = np.genfromtxt("../Label.csv", delimiter=',')
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')

	angles, xs, ys = [], [], []
	for label in labels[:sampleSize]:
		run, step, x, y = label
		run, step = int(run) - 1, int(step) - 1
		prevAngleSeq = observations[run, step-numPrevAngles:step]

		angles.append(prevAngleSeq)
		xs.append(x)
		ys.append(y)

	angles = np.array(angles)
	xs = np.array(xs)
	ys = np.array(ys)

	print "Run SVR to predict X locations"

	xModel = SVR(C=1e3, gamma=0.1)
	xModel.fit(angles, xs)
	joblib.dump(xModel, 'svr_xmodel_%i.pkl' % numPrevAngles) 

	print "Run SVR to predict Y locations"

	yModel = SVR(C=1e3, gamma=0.1)
	yModel.fit(angles, ys)
	joblib.dump(yModel, 'svr_ymodel_%i.pkl' % numPrevAngles) 

# Create submission file using 4000 (x, y) predicted location points
def createSubmission(predLocations, fileName):
	print 'Creating submission file...'

	with open(fileName, 'wb') as f:
		f.write('Id,Value\n')

		for i, (x, y) in enumerate(predLocations):
			xLine = ','.join([str(i + 6001) + 'x', str(x)])
			yLine = ','.join([str(i + 6001) + 'y', str(y)])
			f.write(xLine + '\n')
			f.write(yLine + '\n')

	print 'Done creating submission file...'

if __name__ == '__main__':
	sampleSize = 10000
	numPrevAngles = 5

	# Train SVR using labels

	print "Start training SVR"
	runSVR(sampleSize, numPrevAngles)
	print "Done training SVR"

	# Predict last 4000 1001th locations

	print "Start predicting using SVR"
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	data = observations[6000:,-numPrevAngles:]

	xModel = joblib.load('svr_xmodel_%i.pkl' % numPrevAngles)
	yModel = joblib.load('svr_ymodel_%i.pkl' % numPrevAngles)

	xs = xModel.predict(data)
	ys = yModel.predict(data)

	predLocations = zip(xs, ys)
	createSubmission(predLocations, 'svr_submission_%i.csv' % numPrevAngles)
	print "Done predicting using SVR"



	