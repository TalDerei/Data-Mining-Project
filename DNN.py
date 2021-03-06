
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

class Softmax:
    @staticmethod
    def activate(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis = 0)

class Tanh:
    @staticmethod
    def activate(Z):
        return (np.exp(Z) - np.exp(-1 * Z)) / (np.exp(Z) + np.exp(-1 * Z))

    @staticmethod
    def gradient(Z):
        return (1 - (np.exp(2 * Z) - 2 + np.exp(-2 * Z)) / (np.exp(2 * Z) + 2 + np.exp(-2 * Z)))

class ReLU:
    @staticmethod
    def activate(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def gradient(Z):
        return np.asmatrix(np.where(Z > 0, 1, 0))

class NN:
    def __init__(self, dimensions, activation_funcs):

        self.num_layers = len(dimensions) - 1

        self.W = {}
        self.b = {}
        self.g = {}
        num_neurons = {}
        for l in range(self.num_layers):
            num_neurons[l + 1] = dimensions[l + 1]
            start, end = dimensions[l], dimensions[l + 1]
            self.W[l + 1] = np.random.normal(0.0, .1, (end, start))
            self.b[l + 1] = np.zeros((dimensions[l + 1], 1))
            self.g[l + 1] = activation_funcs[l + 1]

        self.A = {}
        self.Z = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}
    def forward(self, X):
        self.A[0] = X
        for l in range(1, self.num_layers + 1): 
            self.Z[l] = np.dot(self.W[l], self.A[l - 1]) + self.b[l]
            self.A[l] = self.g[l].activate(self.Z[l])
        return self.A[self.num_layers]

    def backward(self, Y):
        l = self.num_layers
        m = Y.shape[1]
        self.dZ[l] = self.A[l] - Y
        self.dW[l] = np.dot(self.dZ[l], self.A[l - 1].T) / m
        self.db[l] = np.sum(self.dZ[l], axis=1) / m
        while l > 1:
           self.dZ[l - 1] = np.multiply(np.dot(self.W[l].T, self.dZ[l]), self.g[l - 1].gradient(self.Z[l - 1]))
           self.dW[l - 1] = np.dot(self.dZ[l - 1], self.A[l - 2].T) / m
           self.db[l - 1] = np.sum(self.dZ[l - 1], axis=1) / m
           l -= 1
        return self.dW, self.dZ

    def update_parameters(self, lr, weight_decay = 0.001):
        for l in range(1, self.num_layers + 1):
            self.W[l] -= self.W[l] * weight_decay + lr * self.dW[l]
            self.b[l] -= lr * self.db[l].reshape(self.b[l].shape)
        
    def train(self, X_train, Y_train, X_test, Y_test, iter_num, lr, weight_decay, record_every):
        for it in range(iter_num):
              self.forward(X_train)
              self.backward(Y_train)
              self.update_parameters(lr, weight_decay)
            # tracking the test error during training.
              if (it + 1) % record_every == 0:
                   prediction_accuracy = self.test(X_test, Y_test)
                   print(', test error = {}'.format(prediction_accuracy))

    def test(self, X_test, Y_test):
        output = self.forward(X_test)
        predicted_labels = np.argmax(output, axis = 0)
        true_labels = np.argmax(Y_test, axis = 0)
        return 1.0 - accuracy_score(np.array(true_labels).flatten(), np.array(predicted_labels).flatten())

if __name__ == "__main__":
    print("Please enter file (e.g. yale.txt, cho.txt, iyer.txt): ")
    fname = input()
    files = ["yale.txt", "cho.txt", "iyer.txt"]
    classes = [38,5,10]
    num_classes = classes[files.index(fname)]
    points = []
    truth = []
    t_points = []
    t_truth = []
    if(fname == "yale.txt"):
        with open("StTrainFile1.txt", "r") as f1:
            red = f1.read().split("\n")
            for i in red:
                i = i.split(" ")
                if(len(i) < 2):
                    continue
                toAdd = []
                totruth = [0 for i in range(num_classes)]
                totruth[int(i[-1]) - 1] = 1
                for j in range(len(i) - 1):
                    toAdd.append((float(i[j]) - .1) * 10)
                points.append(np.array(toAdd))
                truth.append(np.array(totruth))
        with open("StTestFile1.txt", "r") as f2:
            red = f2.read().split("\n")
            for i in red:
                i = i.split(" ")
                if(len(i) < 2):
                    continue
                toAdd = []
                totruth = [0 for i in range(num_classes)]
                totruth[int(i[-1]) - 1] = 1
                for j in range(len(i) - 1):
                    toAdd.append((float(i[j]) - .1) * 10)
                t_points.append(np.array(toAdd))
                t_truth.append(np.array(totruth))
    else:
     with open(fname, "r") as f:
        red = f.read().split("\n")
        counter = 1
        for i in red:
            i = i.split(" ")
            if(len(i) > 1):
                    if(i[1] == -1):
                        continue
                    else:
                        toAdd = []
                        totruth = [0 for i in range(num_classes)]
                        totruth[int(i[1]) - 1] = 1
                        for j in range(2, len(i)):
                            toAdd.append(float(i[j]))
                        if(counter % 5 != 0):
                            truth.append(np.array(totruth))
                            points.append(np.array(toAdd))
                        else:
                            t_truth.append(np.array(totruth))
                            t_points.append(np.array(toAdd))
                        counter += 1   
    tr_X, tr_Y, te_X, te_Y = np.array(points).T, np.array(truth).T, np.array(t_points).T, np.array(t_truth).T

    #Hyperparameters
    iter_num = 1001
    lr = 0.1
    weight_decay = 0.0001
    record_every = 50

    input_dim, n_samples = tr_X.shape
    dimensions = [input_dim, 50, 100, num_classes]
    activation_funcs = [None, Tanh, ReLU, Softmax]
    nn = NN(dimensions, activation_funcs)
    nn.train(tr_X, tr_Y, te_X, te_Y, iter_num, lr, weight_decay, record_every)
    # evaluate the model on the test set and report the detailed performance
    predicted = np.argmax(nn.forward(te_X), axis=0)
    ground_truth_labels = np.argmax(te_Y, axis=0)
    print("Classification report for the neural network \n%s\n"
          % metrics.classification_report(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted), digits=3))
    print("Confusion matrix \n%s\n"
          % metrics.confusion_matrix(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted)))
    print("F1 Score:")
    print(metrics.f1_score(ground_truth_labels, predicted, average='macro'))
    print("AUC:")
    pr, tr, _ = metrics.roc_curve(ground_truth_labels, predicted, pos_label=1)
    print(metrics.auc(pr, tr))

