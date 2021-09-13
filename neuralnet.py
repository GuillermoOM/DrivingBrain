import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from numpy.core.shape_base import _accumulate
from numpy.lib.function_base import _percentile_dispatcher
from numpy.lib.ufunclike import _deprecate_out_named_y

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, i_weight=0.01, i_bias=0.0):
        self.weights = i_weight * np.random.randn(n_inputs, n_neurons)
        self.biases = i_bias * np.random.randn(1, n_neurons)
    
    def inherit_and_evolve_WB(self, old_w, old_b, wmf=0.01, bmf=0.01):
        # old_w inherited weight, old_b inherited bias, wmf weight mutation factor, bmf bias mutation factor
        self.weights = old_w + np.random.randn(self.weights.shape[0], self.weights.shape[1])*wmf
        self.biases = old_b + np.random.randn(1, self.biases.shape[0])*bmf
    
    def inherit_WB(self, old_w, old_b):
        self.weights = old_w
        self.biases = old_b

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CatergoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CatergoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, Learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = Learning_rate
        self.current_learning_rate = Learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)

    # print(X)

    dense1 = Layer_Dense(2, 64)

    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(64, 3)

    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_SGD(Learning_rate=1, decay=1e-3, momentum=0.9)

    # for epoch in range(10001):
    #     dense1.forward(X)
    #     activation1.forward(dense1.output)
    #     dense2.forward(activation1.output)

    #     loss = loss_activation.forward(dense2.output, y)

    #     # print('loss:', loss)

    #     predictions = np.argmax(loss_activation.output, axis=1)
    #     if len(y.shape) == 2:
    #         y = np.argmax(y, axis=1)
    #     accuracy = np.mean(predictions == y)

    #     # print('acc:', accuracy)

    #     if not epoch % 100:
    #         print(f'epoch: {epoch}, ' +
    #               f'acc: {accuracy:.3f}, ' +
    #               f'loss: {loss:.3f}, ' +
    #               f'lr : {optimizer.current_learning_rate}')

    #     loss_activation.backward(loss_activation.output, y)
    #     dense2.backward(loss_activation.dinputs)
    #     activation1.backward(dense2.dinputs)
    #     dense1.backward(activation1.dinputs)

    #     optimizer.pre_update_params()
    #     optimizer.update_params(dense1)
    #     optimizer.update_params(dense2)
    #     optimizer.post_update_params()
