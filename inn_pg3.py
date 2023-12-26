import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x,0)

def sigmoid_derivative(x):
    return x*(1-x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu_derivative(x):
    return np.where(x>0,1,0)


def intializeWeights(input_size, hidden_size, output_size):
    np.random.seed(42)
    hidden_layer_input_weights = np.random.randn(input_size, hidden_size)
    hidden_layer_output_weights = np.random.randn(hidden_size, output_size)

    return hidden_layer_input_weights, hidden_layer_output_weights

def forward_propogation(inputs, hidden_layer_input_weights, hidden_layer_output_weights, activation_function):

    hidden_layer_input = np.dot(inputs,hidden_layer_input_weights)
    hidden_layer_output = activation_function(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output,hidden_layer_output_weights)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output,output_layer_output

def backward_propogation(inputs, target, hidden_layer_input_weights, hidden_layer_output_weights,
                         hidden_layer_output,output_layer_output, learning_rate, activation_derivative):

    output_error = target-output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_delta.dot(hidden_layer_output_weights.T)
    hidden_layer_delta = hidden_layer_error * activation_derivative(hidden_layer_output)

    hidden_layer_output_weights += hidden_layer_output.T.dot(output_delta)*learning_rate
    hidden_layer_input_weights +=  inputs.T.dot(hidden_layer_delta)*learning_rate

def training_neural_network(inputs, target, hidden_size, output_size, learning_rate, ephocs, activation_derivative, activation_function):
    input_size = inputs.shape[1]

    hidden_layer_input_weights, hidden_layer_output_weights = intializeWeights(input_size, hidden_size, output_size)

    loss_history = []

    for epoch in range(ephocs):

        hidden_layer_output,output_layer_output=forward_propogation(inputs, hidden_layer_input_weights, hidden_layer_output_weights, activation_function)

        backward_propogation(inputs, target, hidden_layer_input_weights, hidden_layer_output_weights,
                         hidden_layer_output,output_layer_output, learning_rate, activation_derivative)

        loss= calucate_loss(target, output_layer_output)

        loss_history.append(loss)

        if epoch % 1000 == 0:
            print(f"epoch : {epoch} loss : {loss} ")

    return loss_history,hidden_layer_input_weights, hidden_layer_output_weights

def calucate_loss(target, output_layer_output):
    loss = np.mean((target-output_layer_output)**2)
    return loss


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[1],[1],[0]])
output_size = 1
hidden_size = 4
epochs = 10001
learning_rate = 0.01

sigmoid_loss_history, sigmoid_hidden_layer_input_weights, sigmoid_hidden_layer_output_weights = training_neural_network(inputs, target, hidden_size, output_size,
                                                                                                                        learning_rate, epochs, sigmoid_derivative, sigmoid)

tanh_loss_history, tanh_hidden_layer_input_weights, tanh_hidden_layer_output_weights = training_neural_network(inputs, target, hidden_size, output_size,
                                                                                                                        learning_rate, epochs, tanh_derivative, tanh)

relu_loss_history, relu_hidden_layer_input_weights, relu_hidden_layer_output_weights = training_neural_network(inputs, target, hidden_size, output_size,
                                                                                                                        learning_rate, epochs, relu_derivative, relu)

plt.plot(sigmoid_loss_history, label='sigmoid')
plt.plot(tanh_loss_history, label='tanh')
plt.plot(relu_loss_history, label='relu')
plt.title("Loss History")
plt.xlabel('epochs')
plt.ylabel('loss_history')
plt.legend()
plt.show()
