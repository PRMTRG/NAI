import csv
import random
import numpy
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, n_in, bias, f, f_prim):
        self.n_in = n_in
        self.bias = bias
        self.f = f
        self.f_prim = f_prim
    def initialize_random_weights(self):
        self.weights = [ random.random() for i in range(self.n_in) ]
    def set_input_values(self, input_values):
        self.input_values = input_values
    def calculate_sum(self):
        s = 0
        for i in range(self.n_in):
            s += self.weights[i] * self.input_values[i]
        s += self.bias
        return s
    def calculate_output(self):
        return self.f(self.calculate_sum())
    def error_correction(self, error, eta):
        for i in range(self.n_in):
            delta_w = -eta * self.f_prim(self.calculate_sum()) * error * self.input_values[i]
            self.weights[i] += delta_w


def read_input_data(filename):
    X = []
    Y = []
    file = open(filename, "rt")
    dataset = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    first_line = next(dataset)
    Nin = len(first_line) - 1
    X.append(first_line[0:Nin])
    Y.append(first_line[Nin])
    for line in dataset:
        X.append(line[0:Nin])
        Y.append(line[Nin])
    
    return Nin,X,Y


def initialize_weights(Nin):
    return [random.random() for i in range(Nin)]


def f(x):
    return numpy.tanh(x)


def fprim(x):
    return 1 - (f(x) ** 2)


def train(epochs, X, Y, eta, neuron):  
    RMSE = []
    for epoch in range(epochs):
        RMSE_i = []
        for i in range(len(X)):
            neuron.set_input_values(X[i])
            y_out = neuron.calculate_output()
            error = y_out - Y[i]
            neuron.error_correction(error, eta)
            RMSE_i.append(0.5 * (error ** 2))
        err_sum = 0
        for i in range(len(RMSE_i)):
            err_sum += RMSE_i[i]
        RMSE.append(err_sum / len(RMSE_i))
    
    plt.plot(RMSE)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.show()


def test(filename, neuron):
    Nin, Xtest, Ytest = read_input_data(filename)
    Y = []
    for i in range(len(Xtest)):
        sum_weighted = 0
        for j in range(Nin):
            sum_weighted += neuron.weights[j] * Xtest[i][j]
        Y.append(neuron.f(sum_weighted))
    
    return Y, Ytest


if __name__ == '__main__':    
 
    # Get the train data
    Nin, Xtrain, Ytrain = read_input_data("train_data.csv")
    
    neuron = Neuron(Nin, 0, f, fprim)

    # Initialize weights
    neuron.initialize_random_weights()
    
    # Train of the perceptron
    epochs = 10000
    eta = 0.1
    train(epochs, Xtrain, Ytrain, eta, neuron)
    
    # Test of the perceptron with the trained weights
    Yout, Yexpected = test("test_data.csv", neuron)
    print("Results:         ", [ '%.2f' % num for num in Yout ])
    print("Expected results:", [ '%.2f' % num for num in Yexpected])


