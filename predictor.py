import numpy as np
import scipy
from scipy.optimize import nnls
import csv
import sys
import argparse

class Predictor(object):

  def __init__(self, training_data_in=[], data_file=None):
    ''' 
        Initiliaze the Predictor with some training data
        The training data should be a list of [mcs, input_fraction, time]
    '''
    self.training_data = []
    self.training_data.extend(training_data_in)
    if data_file:
      with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
          if row[0].isdigit():
            depth = int(row[0])
            fanout = int(row[1])
            time = float(row[2])
            self.training_data.append([depth, fanout, time])

  def add(self, mcs, input_fraction, time):
    self.training_data.append([mcs, input_fraction, time])

  def predict(self, input_fraction, mcs):
    ''' 
        Predict running time for given input fraction, number of machines.
    '''    
    test_features = np.array(self._get_features([input_fraction, mcs]))
    return test_features.dot(self.model[0])

  def predict_all(self, test_data):
    ''' 
        Predict running time for a batch of input sizes, machines.
        Input test_data should be a list where every element is (input_fraction, machines)
    '''    
    test_features = np.array([self._get_features([row[0], row[1]]) for row in test_data])
    return test_features.dot(self.model[0])

  def fit(self):
    print("Fitting a model with ", len(self.training_data), " points")
    labels = np.array([row[2] for row in self.training_data])
    data_points = np.array([self._get_features(row) for row in self.training_data])
    self.model = nnls(data_points, labels)
    # TODO: Add a debug logging mode ?
    # print "Residual norm ", self.model[1]
    # print "Model ", self.model[0]
    # Calculate training error
    training_errors = []
    for p in self.training_data:
      predicted = self.predict(p[0], p[1])
      training_errors.append(predicted / p[2])

    training_errors = [str(np.around(i*100, 2)) + "%" for i in training_errors]
    print("Prediction ratios are", ", ".join(training_errors))
    return self.model

  def num_examples(self):
    return len(self.training_data)

  def _get_features(self, training_point):
    mc = training_point[0]
    scale = training_point[1]
    return [1.0, float(scale) / float(mc), float(mc), np.log(mc)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment Design')

  parser.add_argument('--csv-path', type=str, required=True,
      help='The csv files containing the training data')
  
  args = parser.parse_args()
  print("csv file:", args.csv_path)

  pred = Predictor(data_file=args.csv_path)

  print("Model Fitting")
  model = pred.fit()
  print("Solution Vector", len(model[0]))
  
  test_data = [[i, 1.0] for i in range(4, 64, 4)]

  predicted_times = pred.predict_all(test_data)
  print("Depth Fanout Predicted-Time")
  for i in range(0, len(test_data)):
    print(test_data[i][0],test_data[i][1], predicted_times[i])

  # A = np.array([[1, 0,0], [1, 1,0], [0,2, 1]])
  # b = np.array([2, 1, 1])
  # model = nnls(A, b)
  # print(model[0],model[1])

