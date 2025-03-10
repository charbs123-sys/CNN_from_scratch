#import tensorflow.keras as tk
import numpy as np
from scipy.signal import correlate2d
from skimage.util import view_as_windows
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

def cross_correlation(activation, kernel, stride = 1):
    mat_forward = correlate2d(activation, kernel, mode = "valid")
    return mat_forward[::stride, ::stride]

class Layer():
    def __init__(self, stride, size, num_filters):
        self.stride = stride
        self.size = size
        self.num_filters = num_filters
        self.input_channels = None  # Determined during first forward pass
        self.filters = None  # Initialized dynamically
        self.bias = np.zeros(num_filters)
        self.RELU = ReLu()

    def forward(self, input_image):
        # Ensure input is 3D: (H, W, C)
        if input_image.ndim == 2:
            input_image = input_image[..., np.newaxis]  # Add channel dim (C=1)
        self.X = input_image
        H, W, input_channels = input_image.shape

        # Initialize filters if not done yet (based on input_channels)
        if self.filters is None:
            self.input_channels = input_channels
            self.filters = np.random.randn(
                self.num_filters, self.size, self.size, input_channels
            ) * np.sqrt(2 / (self.size * self.size * input_channels))

        outputs = []

        for i in range(self.num_filters):
            # Get i-th filter (shape: (size, size, input_channels))
            filter = self.filters[i]
            # Compute cross-correlation across all channels
            conv_out = np.zeros((
                (H - self.size) // self.stride + 1,
                (W - self.size) // self.stride + 1
            ))

            for c in range(self.input_channels):
                # Cross-correlate each channel and accumulate
                channel_act = input_image[:, :, c]
                channel_kernel = filter[:, :, c]
                conv_out += correlate2d(
                    channel_act, channel_kernel, mode='valid'
                )[::self.stride, ::self.stride]

            # Add bias and apply ReLU
            conv_out += self.bias[i]
            conv_out = self.RELU.relu_forward(conv_out)
            outputs.append(conv_out)
        self.outputs = outputs
        # Stack outputs along the channel dimension (shape: H', W', num_filters)

        #print(np.stack(outputs, axis = -1).shape)
        return np.stack(outputs, axis=-1)

    def backwardprop(self, dL_dA, learning_rate=0.01):

        H, W, C = self.X.shape  # Input shape (H, W, input_channels)
        dL_dX = np.zeros_like(self.X)
        filter_grads = np.zeros_like(self.filters)
        bias_grads = np.zeros_like(self.bias)

        for i in range(self.num_filters):
            # Gradient for the i-th filter's output (shape: H', W')
            dL_dA_filter = dL_dA[:, :, i]

            # ReLU derivative for this filter's output
            dA_dZ = (self.outputs[i] > 0) * 1.0  # (H', W')
            dL_dZ = dL_dA_filter * dA_dZ  # (H', W')

            # Upsample dL_dZ to pre-strided dimensions (H_conv, W_conv)
            H_conv = (H - self.size) // self.stride * self.stride + self.size
            W_conv = (W - self.size) // self.stride * self.stride + self.size
            dL_dZ_upsampled = np.zeros((H_conv, W_conv))
            h_indices = np.arange(0, H_conv, self.stride)[:dL_dZ.shape[0]]
            w_indices = np.arange(0, W_conv, self.stride)[:dL_dZ.shape[1]]
            dL_dZ_upsampled[h_indices[:, None], w_indices] = dL_dZ

            # Compute filter gradient for each input channel
            for c in range(self.input_channels):
                # Input channel c: shape (H, W)
                channel_input = self.X[:, :, c]
                # Valid cross-correlation to compute filter gradient
                filter_grads[i, :, :, c] = correlate2d(
                    channel_input, dL_dZ_upsampled, mode='valid'
                )

            # Compute bias gradient
            bias_grads[i] = np.sum(dL_dZ)

            # Compute input gradient contribution from this filter
            for c in range(self.input_channels):
                flipped_kernel = np.flipud(np.fliplr(self.filters[i, :, :, c]))
                temp = correlate2d(dL_dZ_upsampled, flipped_kernel, mode='full')
                # Calculate the amount to crop: assuming kernel size F
                F = self.size
                start_h = (temp.shape[0] - H) // 2
                start_w = (temp.shape[1] - W) // 2
                dL_dX[:, :, c] += temp[start_h:start_h + H, start_w:start_w + W]

        # Update weights and biases
        self.filters -= learning_rate * filter_grads
        self.bias -= learning_rate * bias_grads
        return dL_dX


class Dense():
    def __init__(self, input_shape):
        # input_shape: (num_kernels, flattened_features, 1) = (10, 25, 1)
        self.input_dim = input_shape[0] * input_shape[1]  # 10 * 25 = 250
        self.output_dim = 10  # MNIST has 10 classes

        # He initialization for weights
        self.W = np.random.randn(self.output_dim, self.input_dim) * np.sqrt(2 / self.input_dim)
        self.b = np.zeros((self.output_dim, 1))  # Bias shape: (10, 1)

    def forward(self, X):
        # X shape: (10, 25, 1) → Flatten to (250, 1)
        self.init_shape = X.shape
        self.X = X.reshape(-1, 1)  # Flatten to (250, 1)
        
        # Compute output: Z = W.X + b
        Z = np.dot(self.W, self.X) + self.b  # Shape: (10, 1)
        return Z.flatten()  # Return 1D array for softmax
        
    def backward(self, softmax, y, learning_rate = 0.01):
        dL_dZ = np.array(softmax.reshape(-1, 1)) - np.array(y.reshape(-1, 1))
        dL_dZ = dL_dZ.reshape(-1,1)
        dL_dW =  np.matmul(dL_dZ, self.X.transpose())
        dL_db = np.sum(dL_dZ, axis = 1, keepdims=True)
        dL_dA = np.matmul(self.W.transpose(), dL_dZ)
        
        self.W -= learning_rate * dL_dW
        self.b -= learning_rate * dL_db
        return dL_dA.reshape(self.init_shape)

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

import numpy as np
from skimage.util import view_as_windows

class Pooling():
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def forward(self, activation):
        pooled_filters = []
        masks = []
        coords_all = []
        #print(activation)
        if len(activation.shape) == 2:
            activation = [activation]
        for i in range(activation.shape[2]):
            #print(activation)
            act = activation[:,:,i]
            h, w = act.shape
            pooled_h = (h - self.size) // self.stride + 1
            pooled_w = (w - self.size) // self.stride + 1

            # Create sliding windows view of the activation array.
            windows = view_as_windows(act, (self.size, self.size), step=self.stride)
            # Compute the pooled output by taking the maximum over the last two dimensions.
            pooled = np.max(windows, axis=(2, 3))

            # Reshape windows to combine the window dimensions so we can get argmax over each window.
            flat_windows = windows.reshape(pooled_h, pooled_w, -1)
            max_idx = np.argmax(flat_windows, axis=2)

            # Calculate row and column offsets within each window.
            offset_rows = max_idx // self.size
            offset_cols = max_idx % self.size

            # Calculate the coordinates in the original activation map.
            row_indices = np.arange(pooled_h)[:, None] * self.stride + offset_rows
            col_indices = np.arange(pooled_w)[None, :] * self.stride + offset_cols

            # Initialize and update the boolean mask for max positions.
            self.mask = np.zeros(act.shape, dtype=bool)
            self.mask[row_indices, col_indices] = True
            masks.append(self.mask)

            # Store coordinates for backward propagation.
            self.coords = list(zip(row_indices.ravel(), col_indices.ravel()))
            coords_all.append(self.coords)
            pooled_filters.append(pooled)
        self.coords = coords_all
        self.mask = np.stack(masks, axis = -1)
        self.pooled = np.stack(pooled_filters, axis = -1)
        pooled_filters = np.stack(pooled_filters, axis = -1)

        #print(pooled_filters.shape)
        return pooled_filters

    
    def backwardprop(self, dL_dX):

        # Initialize gradient matrix (same shape as original activation)
        dL_dA = np.zeros_like(self.mask, dtype=np.float32)
        for i in range(dL_dA.shape[2]):
            # Extract coordinates (rows and cols) from self.coords
            rows, cols = zip(*self.coords[i])  # Unpack into separate row/col arrays
            
            # Flatten the incoming gradient to match the order of coordinates
            if dL_dX.shape[2] == 1:
                flat_grad = dL_dX[i].flatten()
            else:
                flat_grad = dL_dX[:,:,i].flatten()
            # Use advanced indexing to scatter gradients to their positions

            np.add.at(dL_dA[:,:,i], (rows, cols), flat_grad)
            
        return dL_dA
    

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
                    #print("first layer")
                    A_conv1 = self.layers[0].forward(batch_data[j])
                    for conv in range(1, len(self.layers)):
                        A_conv1 = self.layers[conv].forward(A_conv1)


                    Z_flat = []
                    for i in range(A_conv1.shape[2]):
                        Z_flat.append(A_conv1[:,:,i].flatten().reshape(-1, 1))
                    #print(np.array(Z_flat).shape)
                    # Initialize Dense layer once
                    if self.Dense_initialized == 0:
                        self.DENSE = Dense(np.array(Z_flat).shape)
                        self.Dense_initialized = 1
                    
                    Z_dense = self.DENSE.forward(np.array(Z_flat)).flatten()

                    # Softmax and loss
                    prediction = Softmax(Z_dense)
                    one_hot_label = np.zeros(10)
                    one_hot_label[batch_labels[j]] = 1
                    loss = cross_entropy(prediction, one_hot_label)
                    batch_loss += loss

                    # Backward pass
                    dL_dA_dense = self.DENSE.backward(prediction, one_hot_label, 0.01)
                    dL_dX = self.layers[len(self.layers) - 1].backwardprop(dL_dA_dense)
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
        return prediction

        


np.random.seed(5)

example = np.load("MNIST_train.npy")
example_labels = np.load("MNIST_train_labels.npy")

test = np.load("MNIST_test.npy")
test_labels = np.load("MNIST_test_labels.npy")

#example = example[0:10000]
#example_labels = example_labels[0:10000]




conv_network = CNN(example, example_labels)
conv_network.convolution_layer(Layer(1, 3, 32))
conv_network.convolution_layer(Pooling(2, 2))
conv_network.convolution_layer(Layer(1, 3, 64))
conv_network.convolution_layer(Pooling(2, 2))


conv_network.train(1, 260)

correct = 0

prediction = conv_network.predict(test[100], test_labels[100])

for i in range(len(test_labels)):
    prediction = conv_network.predict(test[i], test_labels[i])
    max_pred = np.argmax(np.array(prediction))
    if max_pred == test_labels[i]:
        correct += 1
print(correct/len(test_labels))



