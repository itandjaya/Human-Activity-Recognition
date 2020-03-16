## NN_HAR_main.py
## Neural Network - Human Activity Recognition.
## Main function to initiate and train NN model, and
##   validate the model accuracy.
#####################################################


from import_data import load_data;
from NN_Model import NN_Model, one_hot;
from random import randint;
from numpy import save as np_save, load as np_load;

import numpy as np;
import os;
import matplotlib.pyplot as plt;    #   To plot Cost vs. # of iterations.


def main():

    ## Import and prune the data.
    ## Note that features data has been normalized.
    (X_train, y_train), (X_test, y_test)    =   load_data();

    ## Convert label classification to one-hot format (if not using SparseCategoricalCrossentropy).
    #y_train =   one_hot(    y_train);
    #y_test  =   one_hot(    y_test);
    
    model   =   NN_Model();                 ## Initiate NN Model.

    model.train_NN( X_train,  y_train);     ## Train NN model using training dataset.

    model.plot_training_model();            ## Plot the Train and validation accuracy.

    ## Evaluating test dataset accuracy.
    print(" Test Dataset Accuracy:")
    model.test_accuracy(X_test, y_test);    ## loss: 0.2633 - accuracy: 0.9308.
    
    return 0;

if __name__ == "__main__":  main();
