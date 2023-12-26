import numpy as np

class Neuron:
    def __init__(self,input_size,activation_function="sigmoid"):
        np.random.seed(42)
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

        if(activation_function == 'sigmoid'):
            self.activation_function = self.sigmoid
            self_activation_derivative = self.sigmoid_derivative
        
        elif(activation_function == 'step'):
            self.activation_function = self.step
            self_activation_derivative = self.step_derivative
        
        else:
            raise ValueError("Invalid activation function")
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)
    
    def step(self,x):
        return 1 if x>0 else 0
    
    def step_derivative(self,x):
        return 0
        
    def forward(self,inputs):
        weighted_sum = np.dot(inputs,self.weights)+self.bias
        output = self.activation_function(weighted_sum)
        return output

input_size = 3
inputs = np.random.rand(input_size)

neuron = Neuron(input_size,activation_function="sigmoid")

output=neuron.forward(inputs)

n= f' input : {inputs} \n bias = {neuron.bias}\n weights = {neuron.weights} \n output = {output}'

print(n)
