# CNN_from_scratch

Notes for now
#import tensorflow.keras as tk
import numpy as np
from scipy.signal import correlate2d
""" mnist = tk.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.save("MNIST_test.npy",x_test)
np.save("MNIST_test_labels.npy", y_test) """



class ReLu():
    #need to observe how np.maximum interactors with Z which should be a 3 dimensional matrix
    def relu_forward(self, Z):
        A = np.maximum(Z, 0)
        return A
    
    def relu_backward(self, input):
        grad = input > 0
        return input * grad





class Layer():
    """
    To do:
	forward propogation:
	1 - In forward propogation the input is assumed to be of size (H, W, C) where H is the height, W is the width and C is the number of channels
	2 - Change the initialization of the kernel in the __init__ constructor
	3 - For each channel in the initialized kernel, perform a cross correlation with each channel in the input, sum the results of perfoming this operation
	4 - Change bias in __init__ constructor
	5 - Return the output of all filters

	6 - Work on back propogation later
    """
    def __init__(self):
        self.mat = np.random.randn(3, 3) * np.sqrt(2 / (3 * 3))
        self.initialize = self.mat
        self.bias = np.zeros(1)
        self.RELU = ReLu()
    
    def forward(self, input_image):
        self.X = input_image
        mat_forward = correlate2d(input_image, self.mat, mode = "valid")
        self.Z = mat_forward + self.bias
        A_conv = self.RELU.relu_forward(self.Z)
        return A_conv

    def backwardprop(self, dL_dA, learning_rate = 0.01):
        dA_dZ = (self.Z > 0) * 1
        dL_dZ = dL_dA * dA_dZ
        dL_dF = correlate2d(self.X, dL_dZ, mode = "valid")
        dL_db = np.sum(dL_dZ)
        #Need to compute dL_dX
        dL_dX = dL_dX = correlate2d(dL_dZ, np.flip(np.flip(self.mat, 0), 1), mode="full")

        self.mat -= learning_rate * dL_dF
        self.bias -= learning_rate * dL_db
        return dL_dX


class Dense():
    """
    to do:
	forward propogation
	1 - Nothing needs to be done
    """
    def __init__(self, flattened):
        #use He initialization
        
        self.W = np.random.randn(10, len(flattened)) * np.sqrt(2 / len(flattened))
        self.b = np.zeros((10, 1))
    
    def forward(self, X):
        self.X = X
        Z = np.matmul(self.W, X) + self.b
        return Z
        
    def backward(self, softmax, y, learning_rate = 0.01):
        dL_dZ = np.array(softmax.reshape(-1, 1)) - np.array(y.reshape(-1, 1))
        dL_dZ = dL_dZ.reshape(-1,1)
        dL_dW =  np.matmul(dL_dZ, self.X.transpose())
        dL_db = np.sum(dL_dZ, axis = 1, keepdims=True)
        dL_dA = np.matmul(self.W.transpose(), dL_dZ)
        
        self.W -= learning_rate * dL_dW
        self.b -= learning_rate * dL_db
        
        return dL_dA

def Softmax(Z):
    Z = Z - np.max(Z)
    exp_Z = np.exp(Z)  
    return exp_Z / np.sum(exp_Z)

def cross_entropy(prediction, y):
    '''
    cross entropy loss assuming labels are 0 and 1 in y one-hot encoding vector
    '''
    L = -np.sum(y * np.log(prediction + 1e-9))

    return L


class CNN():
    def __init__(self, dataset, labels):
        self.dataset = dataset / 255.0
        self.labels = labels
        self.samples = dataset.shape[0]
        self.Dense_initialized = 0
        self.layers = []

    def convolution_layer(self, layer):
        self.layers.append(layer)

    def train(self, epochs, batch_size = 100):
	"
	1 - check if the flattening operation is performed correctly (should go from multidimensional matrix into single vector)
	2 - Check how minibatches work in more depth
	"
        
        for epoch in range(epochs):
            #shuffle dataset
            indices = np.arange(self.samples)
            np.random.shuffle(indices)
            shuffled_data = self.dataset[indices]
            shuffled_labels = self.labels[indices]

            #for each mini-batch
            for i in range(0, self.dataset.shape[0], batch_size):
                
                batch_data = shuffled_data[i:i + batch_size]
                batch_labels = shuffled_labels[i:i + batch_size]
                batch_loss = 0             
                
                #for each image
                for j in range(batch_data.shape[0]):
                    # Forward pass
                    A_conv1 = self.layers[0].forward(batch_data[j])
                    for conv in range(1, len(self.layers)):
                        A_conv1 = self.layers[conv].forward(A_conv1)

                    Z_flat = A_conv1.flatten().reshape(-1, 1)

                    # Initialize Dense layer once
                    if self.Dense_initialized == 0:
                        self.DENSE = Dense(Z_flat)
                        self.Dense_initialized = 1
                    
                    Z_dense = self.DENSE.forward(Z_flat).flatten()

                    # Softmax and loss
                    prediction = Softmax(Z_dense)
                    one_hot_label = np.zeros(10)
                    one_hot_label[batch_labels[j]] = 1
                    loss = cross_entropy(prediction, one_hot_label)
                    batch_loss += loss

                    # Backward pass
                    dL_dA_dense = self.DENSE.backward(prediction, one_hot_label, 0.01)
                    dL_dA_conv = dL_dA_dense.reshape(A_conv1.shape)
                    dL_dX = self.layers[len(self.layers) - 1].backwardprop(dL_dA_conv)
                    for i in range(len(self.layers) - 2, 0, -1):
                        dL_dX = self.layers[i].backwardprop(dL_dX)
                minibatch_loss = batch_loss/batch_size
                print(minibatch_loss)
        return None
    
    def predict(self, test, test_label):
        
        for layer in self.layers:
            test = layer.forward(test)
        flatten = test.flatten().reshape(-1, 1)
        dense_layer = self.DENSE.forward(flatten).flatten()
        prediction = Softmax(dense_layer)
        print(prediction)
        print(test_label)

        


np.random.seed(5)

example = np.load("MNIST_train.npy")
example_labels = np.load("MNIST_train_labels.npy")

test = np.load("MNIST_test.npy")
test_labels = np.load("MNIST_train_labels.npy")

example = example[0:10000]
example_labels = example_labels[0:10000]


conv_network = CNN(example, example_labels)
conv_network.convolution_layer(Layer())
conv_network.convolution_layer(Layer())
conv_network.convolution_layer(Layer())
conv_network.train(5, 100)

conv_network.predict(example[976], example_labels[976])




