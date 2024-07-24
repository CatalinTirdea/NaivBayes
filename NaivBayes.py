import math
import random
from sys import argv
import pandas as pd
import numpy as np


"""
This class takes the data and 
assigns it in a dictionary. 
data[i][-1] is the last element that is by convention the class name
After that we "encode it" meaning that we give a unique code 
"""


def class_encoder(data):
    classes = []
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes.append(data[i][-1])
    for i in range(len(classes)):
        for j in range(len(data)):
            if data[j][-1] == classes[i]:
                data[j][-1] = i
    return data


"""
Function that normalises the values using the Manhattan distance

"""


def normalize_data(values):
    valuesum = 0.0
    for value in values:
        valuesum += value

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




data = read_and_process_data(argv[1])
train_data,test_data = split_data(data,0.2)

x_train,y_train = split_labels(train_data)
