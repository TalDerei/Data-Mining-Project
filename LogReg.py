import numpy as np
from scipy.special import xlogy
from sklearn import metrics

def linear(W, X):
    return np.inner(W.T, X.T)

def softmax(Z):

    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)

def dW(Z, X, Y):
    dZ = softmax(Z) - Y
    ans = np.dot(dZ,X.T)
    ans = ans.T
    return ans

def train(X, Y, num_iters, lr):
    W = np.zeros((X.shape[0], Y.shape[0]))
    W = np.array(W, dtype=np.double)
    for i in range(0, num_iters):
      Z = np.array(linear(W, X), dtype=np.float64)
      dw =  lr * dW(Z, X, Y)
      W -= dw
    return W
def test(W, X_test, Y_test):
        Zt = np.array(linear(W, X_test), dtype=np.float64)
        predicted = np.argmax(softmax(Zt), axis=0)
        ground_truth_labels = np.argmax(Y_test, axis=0)
        print("Classification report for Logistic Regression \n%s\n"
            % metrics.classification_report(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted), digits=3))
        print("Confusion matrix \n%s\n"
              % metrics.confusion_matrix(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted)))
        print("F1 Score:")
        print(metrics.f1_score(ground_truth_labels, predicted, average='macro'))
        print("AUC:")
        pr, tr, _ = metrics.roc_curve(ground_truth_labels, predicted, pos_label=1)
        print(metrics.auc(pr, tr))
       
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
                    toAdd.append(float(i[j]))
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
                    toAdd.append(float(i[j]))
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
    iter_num = 201
    lr = 0.001
    tr_X, tr_Y, te_X, te_Y = np.array(points).T, np.array(truth).T, np.array(t_points).T, np.array(t_truth).T
    model = train(tr_X, tr_Y, iter_num, lr)
    test(model, np.array(te_X), np.array(te_Y))