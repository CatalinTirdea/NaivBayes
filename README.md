# Naive Bayes Classifier

This project implements a Naive Bayes classifier in Python. The classifier reads data from a CSV file, processes it, trains a model, and makes predictions on a test set. The implementation includes data normalization, training, and evaluation of the model.

## Requirements

The following Python library is required to run the code:

- `numpy`

You can install it using pip:

```bash
pip install numpy
```

Usage

To run the program, use the following command:

```bash

python3 NaivBayes.py <datafile> <ratio>

    <datafile>: Path to the CSV file containing the dataset.
    <ratio>: Ratio to split the data into training and test sets (must be between 0 and 1).
```
Example Command

```bash

python3 NaivBayes.py pima-indians-diabetes.csv 0.2
```
This command will split the dataset pima-indians-diabetes.csv with 20% of the data used as the test set and 80% as the training set.


Functions Overview:


`normalize_data(values)`
Normalizes the feature values using the Manhattan distance (L1 norm).

`read_and_process_data(datafile)`
Reads and processes the data from a CSV file, normalizing the feature values.

`split_data(data, ratio)`
Splits the data into training and test sets based on the given ratio.

`split_labels(data)`
Separates the features and labels from the dataset.

`calculate_unique_probabilities(y_label)`
Calculates the prior probabilities for each class.


`calculate_conditional_probabilities(x_data, y_data)`
Calculates the conditional probabilities for each feature given each class.


`train_naiv_bayes(x_data, y_data)`
Trains the Naive Bayes model by calculating the unique and conditional probabilities.


`predict_naiv_bayes(unique_prob, conditional_prob, x_test)`\
Predicts the class labels for the test set based on the trained model.

`calculate_model_accuracy(y_pred, y_test)`
Calculates the accuracy of the model's predictions.


`Example`

Here's an example of how to use the classifier:

   ``` bash

pip install numpy

Prepare your dataset in CSV format. The last column should be the class label.

Run the Naive Bayes classifier with your dataset:
```
```bash

python3 NaivBayes.py your_dataset.csv 0.2
```
The program will output the accuracy of the model on the test set.