import numpy as np
import collections


# Convolution Layer
class Conv2d():
    def __init__(self, window_size, channels, neurons):
        """
        :param window_size: Sliding Window Size for convolution
        :param channels: Number of channels
        :param neurons: Number of neurons used for calculation
        """

        self.neurons = neurons
        self.window_size = window_size
        self.kernel = np.random.rand(channels, window_size, window_size, self.neurons)

    def forward(self, input_):
        """
        :param input_: single input
        :return: feature map
        """
        self.input_ = input_
        self.width, self.height, self.channel = self.input_.shape[2], self.input_.shape[1], self.input_.shape[-1]

        self.featureMap = np.zeros(shape=(self.input_.shape[0],
                                          self.width - self.window_size + 1,
                                          self.height - self.window_size + 1, self.neurons))

        # convolution
        for i in range(self.height - self.window_size + 1):
            for j in range(self.width - self.window_size + 1):
                patch = self.input_[:, i:i + self.window_size, j:j + self.window_size, [self.channel - 1]]
                matMul = np.sum(patch * self.kernel, axis=(1, 2))
                self.featureMap[:, i, j, :] = matMul

        return self.featureMap

    def backward(self, output_error, learning_rate):
        """
        :param output_error: information from last layer
        :param learning_rate: Learning rate of NN training
        :return: input_error
        """
        self.weight_error = np.zeros(self.kernel.shape)
        input_error = np.zeros(self.input_.shape)

        h, w = self.kernel.shape[1], self.kernel.shape[2]

        # update weight
        for i in range(h):
            for j in range(w):
                weight_obj = self.input_[:, i:self.height + i - self.window_size + 1,
                             j:self.width + j - self.window_size + 1, :]
                product = weight_obj * output_error
                self.weight_error[:, i, j, :] = np.sum(product, axis=(0, 1, 2))

        # update input error
        for i in range(output_error.shape[1]):
            for j in range(output_error.shape[2]):
                input_error[:, i:i + self.window_size, j:j + self.window_size, :] += np.sum(
                    self.kernel * output_error[:, [[i]], [[j]], :], axis=3, keepdims=True)

        self.kernel -= self.weight_error * learning_rate

        return input_error


# Pooling Layer
class MaxPool():
    def __init__(self, stride):
        self.stride = stride

    def forward(self, featureMap):
        self.featureMap = featureMap
        self.h, self.w = self.featureMap.shape[1], self.featureMap.shape[2]
        self.max_output = np.zeros(shape=(self.featureMap.shape[0], self.stride,
                                          self.stride, self.featureMap.shape[-1]))
        self.length = self.w // self.stride

        for i in range(0, self.h, self.length):
            for j in range(0, self.w, self.length):
                self.max_output[:, i // self.length, j // self.length, :] = np.max(
                    self.featureMap[:, i:i + self.length, j:j + self.length, :], axis=(1, 2))

        return self.max_output

    def backward(self, output_error, learning_rate):
        input_error = np.zeros(shape=self.featureMap.shape)

        h, w = self.max_output.shape[1], self.max_output.shape[2]

        for i in range(h):
            for j in range(w):
                original_coordinate = [i * self.length, (i + 1) * self.length, j * self.length, (j + 1) * self.length]
                prev_slice = self.featureMap[:, original_coordinate[0]:original_coordinate[1],
                             original_coordinate[2]:original_coordinate[3], :]
                mask = (prev_slice == np.max(prev_slice, axis=(1, 2), keepdims=True))
                input_error[:, original_coordinate[0]:original_coordinate[1],
                original_coordinate[2]:original_coordinate[3], :] += mask * output_error[:, [[i]], [[j]], :]

        return input_error


# Fully Connect Layer
class FC():
    def __init__(self, input_size, output_size):
        self.w = np.random.normal(0, 1, [input_size, output_size])
        self.b = np.random.normal(0, 1, [1, output_size])

    def forward(self, x):
        self.original_input = x
        self.input = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

        self.output = np.dot(self.input, self.w) + self.b

        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.w.T)
        weight_error = np.dot(self.input.T, output_error)

        self.w -= learning_rate * weight_error
        self.b -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        input_error = np.reshape(input_error, (input_error.shape[0], *(self.original_input.shape[1:])))

        return input_error


def softmax(output):
    logits_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)


def crossEntropyLoss(probs, y_onehot):
    eps = 1e-10
    log_likelihood = -np.log(probs[range(len(probs)), np.argmax(y_onehot, axis=1)] + eps)
    loss = np.sum(log_likelihood) / len(log_likelihood)
    return loss


def crossEntropy(probs, y_onehot):
    return (probs - y_onehot) / len(probs)


class CNN(object):
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, x, y, epochs, batch_size, learning_rate):
        loss_history = []
        num_samples = x.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0
            shuffle_indices = np.random.permutation(num_samples)
            x_shuffle = x[shuffle_indices]
            y_shuffle = y[shuffle_indices]

            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size

                x_batch = x_shuffle[start:end]
                y_batch = y_shuffle[start:end]

                output = x_batch
                layer_output = {'input': output}

                for layer in self.layers:
                    output = layer.forward(output)
                    layer_output[layer.__class__.__name__] = output
                softmax_output = softmax(output)
                layer_output['softmax'] = softmax_output

                loss = self.loss(softmax_output, y_batch)
                error = self.loss_derivative(softmax_output, y_batch)

                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                epoch_loss += loss

            average_loss = epoch_loss / num_batches
            loss_history.append(average_loss)
            print(f"Epoch {epoch + 1}: Loss = {average_loss}")

        return loss_history

    def eval(self, x):
        output = self.forward(x)
        return np.argmax(softmax(output), axis=1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data for demonstration purposes.
x_data = np.random.rand(1000, 64, 64, 1)  # 1000 samples of 32x32 RGB images
y_data = np.random.randint(0, 10, size=(1000, 1))  # 10 classes, labels between 0 and 9

# One-hot encoding labels
y_data_onehot = np.zeros((y_data.size, y_data.max() + 1))
y_data_onehot[np.arange(y_data.size), y_data.flatten()] = 1

# Split the data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_onehot, test_size=0.2, random_state=42)

# Initialize the model
model = CNN()

# Add layers
model.add(Conv2d(window_size=3, channels=1, neurons=30))
model.add(MaxPool(stride=2))
model.add(FC(input_size=120, output_size=10))

# Set loss functions
model.set_loss(crossEntropyLoss, crossEntropy)

# Train the model
loss_history = model.train(x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001)

# Evaluate on test data
y_pred = model.eval(x_test)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
