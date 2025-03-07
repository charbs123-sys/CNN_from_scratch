#import tensorflow.keras as tk
import numpy as np
from scipy.signal import correlate2d
""" mnist = tk.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.save("MNIST_test.npy",x_test)
np.save("MNIST_test_labels.npy", y_test) """



class ReLu():
    
    def relu_forward(self, Z):
        A = np.maximum(Z, 0)
        return A
    
    def relu_backward(self, input):
        grad = input > 0
        return input * grad

class Layer():
    
    def __init__(self, input_image):
        self.mat = np.random.randn(3, 3) * np.sqrt(2 / (3 * 3))
        self.initialize = self.mat
        self.bias = np.zeros(1)
    
    def forward(self, input_image):
        self.X = input_image
        mat_forward = correlate2d(input_image, self.mat, mode = "valid")
        self.Z = mat_forward + self.bias
        return self.Z

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
    def __init__(self, flattened):
        #use He initialization
        
        self.W = np.random.randn(10, len(flattened)) * np.sqrt(2 / len(flattened))
        print(self.W)
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
    
    def train(self, epochs, batch_size = 100):
        self.conv_layer = Layer(self.dataset[0])
        self.conv_layer1 = Layer(self.dataset[0])
        self.RELU = ReLu()
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
                    Z_conv = self.conv_layer.forward(batch_data[j])
                    A_conv = self.RELU.relu_forward(Z_conv)
                    Z_flat = A_conv.flatten().reshape(-1, 1)

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
                    dL_dA_conv = dL_dA_dense.reshape(A_conv.shape)
                    self.conv_layer.backwardprop(dL_dA_conv)
                minibatch_loss = batch_loss/batch_size
                print(minibatch_loss)
        return None
    
    def predict(self, test, test_label):
        convolution = self.conv_layer.forward(test)
        activation = self.RELU.relu_forward(convolution)
        flatten = activation.flatten().reshape(-1, 1)
        dense_layer = self.DENSE.forward(flatten).flatten()
        prediction = Softmax(dense_layer)
        print(prediction)
        print(test_label)

        


np.random.seed(5)

example = np.load("MNIST_train.npy")
example_labels = np.load("MNIST_train_labels.npy")

test = np.load("MNIST_test.npy")
test_labels = np.load("MNIST_train_labels.npy")

example = example[0:100]
example_labels = example_labels[0:100]

conv_network = CNN(example, example_labels)
conv_network.train(5, 100)

conv_network.predict(example[50], example_labels[50])




