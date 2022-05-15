# KNN: Supervised Classification Algorithm 

import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from statistics import mode

class KNN:
    # Default constructor
    def __init__(self, K):
        self.K = K  # number of nearest neighbors (hypertuning parameter)                      

    # Run k-nearest-neigbor algorithm
    def run_KNN(self, user_input):
        self.user_input = user_input

        # Clean dataset and partition into train/test sets
        self.label()

        # Train datasets
        train_results = self.train()

        # Calculate accuracy, f1-score, and AUC
        self.measurments(train_results)

    def label(self): 
        if (self.user_input == 'cho.txt' or self.user_input == 'iyer.txt'):
            if (self.user_input == 'cho.txt'):
                # set K 
                self.K = 7
            if (self.user_input == 'iyer.txt'):
                # set K 
                self.K = 8
            
            # Read dataset
            self.data = pd.read_csv(self.user_input, header = None, delimiter = " ")

            # Partition class labels
            self.class_labels = []
            for i in range(self.data[0].size):
                if (self.data[1][i] == -1):
                    continue
                self.class_labels.append(self.data[1][i])
            self.class_labels = np.array(self.class_labels) 

            # Clean dataset w/o class labels
            for i in range(self.data[0].size):
                if (self.data[1][i] == -1):
                    self.data.drop(i,axis=0,inplace=True)
            self.data.pop(self.data.columns[0])
            self.data.pop(self.data.columns[0])
            self.data_set = self.data.to_numpy()
                    
            # Split into training and test set
            self.training_set, self.testing_set, self.training_labels, self.tesing_labels = train_test_split(self.data_set, self.class_labels, test_size = 0.2, random_state=42)

            # Print training and testing data sets and class labels
            print("Training Dataset: ")
            print(self.training_set,"\n")
            print("Testing Dataset: ")
            print(self.testing_set,"\n")
            print("Training Class Labels: ")
            print(self.training_labels,"\n")
            print("Testing Class Labels: ")
            print(self.tesing_labels,"\n")
        elif (self.user_input == 'yale.txt'):
            # set K 
            self.K = 4

            # Partition training dataset 
            self.training_set = pd.read_csv("StTrainFile1.txt", header = None, delimiter = " ")
            self.training_labels = []
            for i in range(self.training_set[0].size):
                self.training_labels.append(self.training_set[1024][i])
            self.training_labels = np.array(self.training_labels) 
            
            print("Training Class Labels: ")
            print(self.training_labels,"\n")
            print("Training Dataset: ")
            self.training_set = np.array(self.training_set) 
            print(self.training_set,"\n")

            self.testing_set = pd.read_csv("StTestFile1.txt", header = None, delimiter = " ")
            self.tesing_labels = []
            for i in range(self.testing_set[0].size):
                self.tesing_labels.append(self.testing_set[1024][i])
                
            print("Testing Class Labels: ")
            self.tesing_labels = np.array(self.tesing_labels) 
            print(self.tesing_labels,"\n")
            print("Testing Dataset: ")
            self.testing_set = np.array(self.testing_set) 
            print(self.testing_set,"\n")
        else:
            print("Enter valid filename and try again!")
            exit(0)
        
    def train(self): 
        self.predicition = [self.classify(x) for x in self.testing_set]
        return np.array(self.predicition)

    def classify(self, x):
        #compute distances and get k nearest samples
        min_distance = [np.sqrt(np.sum((x - training_set)**2)) for training_set in self.training_set]
        indices = np.argsort(min_distance)[:self.K]
        labels = [self.training_labels[i] for i in indices]
        popular_label = mode(labels)
        return(popular_label)

    def measurments(self, las):
        # Print accuracies, f1-score, and AUC
        print("Classification report for the k-nearest neighbor \n%s\n"
                % metrics.classification_report(np.squeeze(np.asarray(self.tesing_labels)), np.squeeze(self.predicition), digits=3))

def main(): 
    # Call constructor and create object
    obj = KNN(0)

    # Prompt user input
    user_input = input("Please enter file (e.g. yale.txt, cho.txt, iyer.txt): ")

    # Run kmeans algorithm
    obj.run_KNN(user_input)

# Define special variable to execute main function
if __name__== "__main__": 
    main()