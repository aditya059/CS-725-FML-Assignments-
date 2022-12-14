import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''

		"""
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
		"""

		self.layers = [NUM_FEATS] + [num_units] * num_layers + [1] 
		 
		W = []
		b = []
		for i in range(len(self.layers) - 1):
			weights = np.random.uniform(-1, +1, (self.layers[i], self.layers[i + 1]))
			W.append(weights)
			biases = np.random.uniform(-1, +1, (1, self.layers[i + 1]))
			b.append(biases)
		self.weights = W
		self.biases = b

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		def ReLU(z):
			return np.maximum(0, z)

		def ReLU_prime(z):
			z[z <= 0] = 0
			z[z > 0] = 1
			return z 

		A = []
		dz = []
		activations = X

		for i in range(len(self.layers) - 2):
			z = np.matmul(activations, self.weights[i]) + self.biases[i]
			activations = ReLU(z)
			A.append(activations)
			dz.append(ReLU_prime(z))

		self.A = A
		self.dz = dz

		y = np.matmul(activations, self.weights[-1]) + self.biases[-1]
		self.pred = y
		return y
		

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		delta = np.divide(self.pred - y, y.shape[0])
		del_W = []
		del_b = []

		for i in reversed(range(len(self.weights) - 1)):
			del_W.append((np.matmul(self.A[i].transpose(), delta) + lamda * self.weights[i + 1]) / y.shape[0])
			del_b.append(np.mean(delta, axis = 0) + lamda * self.biases[i + 1] / y.shape[0])
			delta = np.matmul(delta, self.weights[i + 1].transpose()) * self.dz[i]

		del_W.append((np.matmul(X.transpose(), delta) + lamda * self.weights[0]) / y.shape[0])
		del_b.append(np.mean(delta, axis = 0) + lamda * self.biases[0] / y.shape[0])
		del_W.reverse()
		del_b.reverse()

		return del_W, del_b

class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		for i in range(len(weights)):
			weights[i] = weights[i] - (self.learning_rate * delta_weights[i])
			biases[i] = biases[i] - (self.learning_rate * delta_biases[i])
		
		return weights,biases


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	return np.mean((y_hat - y) ** 2)

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	loss_reg = 0.0

	for w, b in zip(weights, biases):
		loss_reg += np.linalg.norm(w) ** 2 + np.linalg.norm(b) ** 2

	return loss_reg

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	return (loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases))

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	return np.sqrt(np.mean((y_hat - y) ** 2))


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			#print(e, i, rmse(batch_target, pred), batch_loss)

		print(e, epoch_loss)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	
	
	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print('RMSE on dev data: {:.5f}'.format(dev_rmse))

	train_pred = net(train_input)
	train_rmse = rmse(train_target, train_pred)

	print('RMSE on train data: {:.5f}'.format(train_rmse))

	


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	predicted_y = net(inputs)
	dataset = pd.DataFrame({'Id': np.arange(1.0, predicted_y.shape[0] + 1.0, 1.0), 'Predicted': np.round(predicted_y[:, 0])})
	dataset.to_csv('203050003.csv',index=False, header=True)


def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	# Read train dataset
	df = pd.read_csv('dataset/train.csv')
	train_target = df[list(df.columns)[0]].to_numpy().reshape(df.shape[0],1)
	df.drop(list(df.columns)[0], axis='columns', inplace=True)
	train_input = df.to_numpy()

	# Read dev dataset
	df = pd.read_csv('dataset/dev.csv')
	dev_target = df[list(df.columns)[0]].to_numpy().reshape(df.shape[0],1)
	df.drop(list(df.columns)[0], axis='columns', inplace=True)
	dev_input = df.to_numpy()

	# Read test dataset
	df = pd.read_csv('dataset/test.csv')
	test_input = df.to_numpy()
	
	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128


	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()

