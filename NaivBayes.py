import math
import random
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
