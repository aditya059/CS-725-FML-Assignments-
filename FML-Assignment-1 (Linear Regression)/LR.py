import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        pass
    def __call__(self,features, is_train=False):
        if is_train:
            self.fit_transform(features)
        else:
            self.transform(features)
    def transform(self,features):
        return (features - self.mean) / self.std_dev
    def fit_transform(self,features):
        self.mean = np.mean(features, axis = 0)
        self.std_dev = np.std(features, axis = 0)
        return self.transform(features)
        
def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    df = pd.read_csv(csv_path)
    if(list(df.columns)[-1].strip() == 'shares'):
        df.drop(list(df.columns)[-1], axis='columns', inplace=True)
    X = df.to_numpy(dtype ='float32')
    if scaler != None:
        if is_train:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        
    temp = np.ones((X.shape[0], X.shape[1] + 1))
    temp[:,:-1] = X
    X = temp
    return X

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''

    df = pd.read_csv(csv_path)
    df.drop(list(df.columns)[:-1], axis='columns', inplace=True)
    y = df.to_numpy(dtype ='float32')
    return y
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    
    X = feature_matrix
    X_transpose = X.transpose()
    y = targets
    I = np.identity(X_transpose.shape[0])
    w = np.matmul(np.linalg.inv(np.matmul(X_transpose,X) + (C * I)), np.matmul(X_transpose,y))
    return w

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    predicted_y = np.dot(feature_matrix, weights)
    return predicted_y

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''

    y = targets
    X = feature_matrix
    w = weights
    loss_mse = np.mean((np.dot(X, w) - y) ** 2)
    return loss_mse

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    l2_reg = np.linalg.norm(weights[:-1]) ** 2
    return l2_reg

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    return (mse_loss(feature_matrix, weights,targets) + C * l2_regularizer(weights))


def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    X = feature_matrix
    X_transpose = X.transpose()
    y = targets
    w = weights
    part1 = np.matmul(np.dot(X_transpose, X),w)
    part2 = np.matmul(X_transpose, y) 
    gradient = np.divide(part1 - part2, y.shape[0]) + (C * w)
    return 2 * gradient


def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    random_numbers = np.random.choice(feature_matrix.shape[0], size = batch_size, replace=False)
    a = feature_matrix[random_numbers,:]
    b = targets[random_numbers,:]
    return a,b
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.zeros([n,1])
    #return np.random.rand(n,1)

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    
    return weights - np.dot(lr,gradients)

def early_stopping(prev_dev_loss, prev_train_loss, current_dev_loss, current_train_loss, persistence_value):
    # allowed to modify argument list as per your need
    # return True or False
    
    dev_loss_slope = (current_dev_loss - prev_dev_loss) / persistence_value
    train_loss_slope = (current_train_loss - prev_train_loss) / persistence_value

    if dev_loss_slope >= 0 or train_loss_slope >= 0:
        return True
    return False
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    n = train_feature_matrix.shape[1]
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))

    prev_dev_loss = dev_loss
    prev_train_loss = train_loss
    persistence_value = 25000

    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''
        if step%persistence_value == 0:
            current_dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            current_train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            if early_stopping(prev_dev_loss, prev_train_loss, current_dev_loss, current_train_loss, persistence_value):
                break
            prev_dev_loss = current_dev_loss
            prev_train_loss = current_train_loss

    #print('lr : {}, C : {} '.format(lr, C))	 # Comment it later
    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)

    return loss

if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-3)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.0004,
                        C=0.0,
                        batch_size=32,
                        max_steps=2000000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    
    
    