from sys import argv
import numpy as np


"""
Function that normalises the values using the Manhattan distance(L1)

"""


def normalize_data(values):
    valuesum = np.sum(values)

    normalised_values = []

    for element in values:
        normalised_values.append(element / valuesum)

    
    return normalised_values


"""
This class takes all the information from the file
"""


def read_and_process_data(datafile):
    data = []
    file = open(datafile, 'r')
    for line in file:
        if line[0] == "#":  # This means that the line is a comment and should be ignored
            continue

        values = line.split(",")     # separate the values in an array
        if values:
            # turn the strings into numbers
            values = [eval(i) for i in values]
            # we normalize the data except the last element, the last element is the class
            normalized_values = normalize_data(values[:len(values)-1])
            # add the class after the normalization
        normalized_values.append(values[-1])

        data.append(normalized_values)

    return data

"""
This functions takes the data and a ratio and splits it.
Ratio must be between 0 and 1.
The smaller the ratio, the more train data you will have
"""
def split_data(data,ratio):
    # If value is not between(0,1) throw error
    if ratio <= 0 and ratio >=1:
        assert("Value must be between (0,1)")
    
    # Calculate the index
    index = int(len(data) * ratio)
    # Split the data 
    train_data = data[index:]
    test_data = data[:index]
    
    return train_data,test_data

"""
Function that takes the data and splits the labels 
"""
def split_labels(data):
    # Take all the data exept the last element on each row
    x_data = [row[:-1] for row in data]
    # Take only the last element of each row
    y_data = [row[-1] for row in data]
    
    return x_data,y_data

"""
Function that calculates the unique probabilities{ Ex.: P(A) }
"""
def calculate_unique_probabilities(y_label):
    # Get all the classes and how many times they appear
    classes,counts = np.unique(y_label,return_counts=True)
    # Calculate the probabilities
    probabilities = counts / len(y_label)
    # Return a dictionary in the form {Class: Probability}
    return dict(zip(classes,probabilities))


"""
Function that calculates conditional probabilities{ Ex.:P(A|B) }
"""
import numpy as np

def calculate_conditional_probabilities(x_data, y_data):
    # Ensure x_data and y_data are numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Initialize conditional probabilities dictionary
    conditional_probabilities = {}
    # Get all the unique classes
    classes = np.unique(y_data)
    # Iterate over all unique classes
    for cls in classes:
        # Get all the positions of the class in y_data
        cls_indices = np.where(y_data == cls)[0]  # Extract the indices array from the tuple
        # Extract all the features of the class from x_data
        x_cls = x_data[cls_indices, :]
        # Create the instance of cls in dictionary
        conditional_probabilities[cls] = {}
        # Iterate through each feature (column)
        for i in range(x_data.shape[1]):
            # Extract all the unique features and their counts
            feature_values, counts = np.unique(x_cls[:, i], return_counts=True)
            # Create a dictionary with the feature_values and their probabilities
            conditional_probabilities[cls][i] = dict(zip(feature_values, counts / len(x_cls)))

    return conditional_probabilities


"""
Function that trains the model
"""

def train_naiv_bayes(x_data,y_data):
    unique_prob = calculate_unique_probabilities(y_data)
    cond_prob = calculate_conditional_probabilities(x_data,y_data)
    return unique_prob,cond_prob



"""
The main function that predicts, given a test set
"""

def predict_naiv_bayes(unique_prob,conditional_prob,x_test):
    # initialize the predictions list
    y_pred = []
    # iterate trough every instance
    for x in x_test:
        # initialize posterior probabilities list
        posteriors = {}
        # iterate trough every class of unique_prob
        for cls in unique_prob:
            # initialize posterior with prior probability
            posterior = unique_prob[cls]
            # iterate over every feature 
            for i, feature_value in enumerate(x):
                # update the posterior value with condtional probability of the feature
                if feature_value in conditional_prob[cls][i]:
                    posterior *= conditional_prob[cls][i][feature_value]
                else:
                    posterior *= 1e-6  # smoothing for unseen values
            # store posterior probability for the current class
            posteriors[cls] = posterior
            # append the class with the maximum probability
        y_pred.append(max(posteriors,key=posteriors.get))
    return np.array(y_pred)



"""
Function that calculates accuracy of the prediciton
"""

def calculate_model_accuracy(y_pred,y_test):
    # check how many predictions are correct
    accuracy = np.mean(y_pred == y_test)
    return accuracy




def main():
    if len(argv) !=  3:
        print("Usage: python3 NaivBayes.py data_set ratio")
        print("Ratio divides the data set. The higher the ratio the more test data you will have")
        return
    data = read_and_process_data(argv[1])

    train_data,test_data = split_data(data,float(argv[2]))

    x_train,y_train = split_labels(train_data)
    x_test,y_test = split_labels(test_data)

    unique_prob,cond_prob = train_naiv_bayes(x_train,y_train)
    y_pred = predict_naiv_bayes(unique_prob,cond_prob,x_test)
    accuracy = calculate_model_accuracy(y_pred,y_test)
    print(f"Model accuracy: {accuracy}")


main()

