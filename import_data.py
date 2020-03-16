"""
## import_data.py 
## imports and pre-process raw data.

Data by:
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz
A Public Domain Dataset for Human Activity Recognition Using Smartphones
21th European Symposium on Artificial Neural Networks, Computational Intelligence and
Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Attribute Information:

For each record in the dataset it is provided:
- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope.
- A 561-feature vector with time and frequency domain variables.
- Its activity label.
- An identifier of the subject who carried out the experiment.

LABELS Classification:
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING

"""

import os
import numpy as np
from random import shuffle;


DATA_PATH           =   os.getcwd() + r'/data/';
X_test_file         =   DATA_PATH + r'X_test.txt';
y_test_file         =   DATA_PATH + r'y_test.txt';   
X_train_file        =   DATA_PATH + r'X_train.txt';
y_train_file        =   DATA_PATH + r'y_train.txt';

def parse_data_file( filename):

    if not filename:        return;

    processed_data  =   [];
    with open( filename, 'r') as fo:

        data_line       =   fo.readline();

        while data_line:
            data_line    =   data_line.split(" ");
            data_line    =   [ float(f)      for f in data_line       if f];

            processed_data.append(  data_line);
            data_line   =   fo.readline();

    return processed_data;

def shuffle_dataset( X, y):
    dataset = [ (feat, label) for feat, label in zip(X, y)];
    shuffle(dataset);
    X, y = (zip(*dataset));

    return np.array(X, dtype = np.float64),    np.array(y, dtype = np.int32);

def load_data( pathfiles = DATA_PATH):

    X_train     =   np.array(   parse_data_file(    X_train_file), dtype = np.float64);
    y_train     =   np.array(   parse_data_file(    y_train_file), dtype = np.int32).flatten()-1;

    X_train, y_train  =   shuffle_dataset(X_train, y_train);


    X_test      =   np.array(   parse_data_file(    X_test_file), dtype = np.float64);
    y_test      =   np.array(   parse_data_file(    y_test_file), dtype = np.int32).flatten()-1;

    X_test, y_test  =   shuffle_dataset(X_test, y_test);

    #print(X_train.shape, y_train.shape);
    #print(X_test.shape, y_test.shape);

    return (X_train, y_train), (X_test, y_test);