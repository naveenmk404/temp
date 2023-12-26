import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs = 1000):
        np.random.seed(20)
        self.weights = np.random.randn(input_size+1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors = []

    def activation_function(self,x):
        return 1 if x>=0 else 0

    def predict(self,inputs):
        weighted_sum = np.dot(self.weights[1:], inputs)+self.weights[0]
        return self.activation_function(weighted_sum)
    
    def train(self,training_data, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for inputs,label in zip(training_data,labels):
                prediction = self.predict(inputs)
                error = label-prediction
                total_error += int(0!=error)

                self.weights[1:] += self.learning_rate * inputs * error
                self.weights [0] +=self.learning_rate * error

            self.errors.append(total_error)
            if epoch % 10 == 0 :
                print(f"epoch : {epoch}, error : {total_error}")

            if total_error == 0:
                print("converged early")
                break

training_data = np.array([[0,0],[0,1],[1,0],[1,1]])
lables = ([0,1,1,1])

perceptron = Perceptron(2)
perceptron.train(training_data,lables)

plt.scatter(training_data[:,0],training_data[:,1], c=lables, cmap='viridis', marker='o')
x_vals = np.linspace(-0.5,1.5,100)
y_vals = (-perceptron.weights[1] * x_vals - perceptron.weights[0])/perceptron.weights[2]
plt.plot(x_vals,y_vals,'r--',label='decesion bounday')
plt.xlabel('input1')
plt.xlabel('input2')
plt.legend()
plt.show()
