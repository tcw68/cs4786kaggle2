from sklearn.svm import SVR
from sklearn.externals import joblib
import numpy as np

def runSVR():
	labels = np.genfromtxt("../Label.csv", delimiter=',')
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')

	angles, xs, ys = [], [], []
	for label in labels[:10000]:
		run, step, x, y = label
		run, step = int(run), int(step)

		prevAngleSeq = observations[run-1,step-31:step-1] # Previous angles
		angles.append(prevAngleSeq)
		xs.append(x)
		ys.append(y)

	angles = np.array(angles)
	xs = np.array(xs)
	ys = np.array(ys)

	print "Run SVR to predict X locations"

	xModel = SVR(kernel='rbf', C=1e3, gamma=0.1)
	xModel.fit(angles, xs)

	joblib.dump(xModel, 'svr_xmodel_last30angles.pkl') 

	print "RUN SVR to predict Y locations"

	yModel = SVR(kernel='rbf', C=1e3, gamma=0.1)
	yModel.fit(angles, ys)

	joblib.dump(yModel, 'svr_ymodel_last30angles.pkl') 

# Create submission file using 4000 (x, y) predicted location points
def createSubmission(predLocations, fileName):
	print 'creating submission file...'
	with open(fileName, 'wb') as f:
		f.write('Id,Value\n')

		for i, (x, y) in enumerate(predLocations):
			xLine = ','.join([str(i+6001)+'x', str(x)])
			yLine = ','.join([str(i+6001)+'y', str(y)])
			f.write(xLine + '\n')
			f.write(yLine + '\n')

if __name__ == '__main__':
	observations = np.genfromtxt("../Observations.csv", delimiter = ',')
	data = observations[6000:,-30:]

	xModel = joblib.load('svr_xmodel_last30angles.pkl')
	yModel = joblib.load('svr_ymodel_last30angles.pkl')

	xs = xModel.predict(data)
	ys = yModel.predict(data)

	predLocations = zip(xs, ys)
	createSubmission(predLocations, 'svr_last30angles_submission.csv')



	