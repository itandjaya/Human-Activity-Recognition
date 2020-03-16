# Human-Activity-Recognition
Human Activity Recognition Using Smartphones sensor signals (accelerator and gyroscope) 

Included files:
-. import_data.py:  A script to import, shuffle, and prune the dataset. Data must be downloaded first. See DATA INFORMATION for details.

-. NN_HAR_main.py:  Main function to initiate and train NN model, and validate the model accuracy.

-. NN_Model.py:     Neural Network Model class.
   
   API functions:
    
    -. train_NN(X, y):          Train the NN model with given dataset X, y.
    
    -. test_accuracy( X, y):    Calculate the accuracy of the Model of the test dataset X, y.
    
    -. plot_training_model():   Plot the accuracy vs. epoch.
    
    -. predict_funct(X):        Predict the label of given X data.

-. ACCURACY vs EPOCH.png: A plot showing train and test accuracy per epoch.

    
DATA INFORMATION:

LABELS Classification:
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING

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

