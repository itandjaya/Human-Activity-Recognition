## NN_Model.py
## Neural Network Model class.

'''
 API functions:
    -. train_NN(X, y):          Train the NN model with given dataset X, y.
    -. test_accuracy( X, y):    Calculate the accuracy of the Model of the test dataset X, y.
    -. plot_training_model():   Plot the accuracy vs. epoch.
    -. predict_funct(X):        Predict the label of given X data.

REVISIONS:
v1.0: Initial release. loss: 0.2633 - accuracy: 0.9308.

'''


from __future__ import division, absolute_import, print_function, unicode_literals;

#import os;
import tensorflow as tf;

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Softmax;
from tensorflow.keras import Model;
from tensorflow.keras.regularizers import l1 as l1, l2 as l2;

import matplotlib.pyplot as plt;    #   To display image/plot.
import numpy as np;

EPOCHS      =   100;


## Inherits tf.keras.Model
class   NN_Model(Model):

    def __init__(self, features_samples = []):
        ## Input: set_norm_params

        super(NN_Model, self).__init__(); 

        self.epochs         =   EPOCHS;
        self.train_history  =   [];

        ## Initialized Neural Network model using Keras Sequential.
        ## 3-Layers NN: [128, 64, 6, 10].
        self.model = tf.keras.models.Sequential([

                            Flatten(input_shape = (561,1)),
                            Dropout(0.25),
                            Dense(128, activation = 'relu'),
                            #BatchNormalization(),

                            Dense(64, activation = 'relu'),
                            BatchNormalization(),

                            Dense(6),
        ]);

        self.compile_model();

        self.model.summary();   #display the architecture of NN model. 

        return;

    def compile_model(self):
        ## Loss function: Use cross-entropy (log).
        loss_fn     =   tf.keras.losses.SparseCategoricalCrossentropy(    from_logits = True);

        ## Optimizer.
        #opt    =  tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False);
        #opt         =   tf.keras.optimizers.SGD(learning_rate=0.01);
        opt         =  tf.keras.optimizers.Adam();

        self.model.compile(     optimizer   =   opt,
                                loss        =   loss_fn,
                                metrics     =   ['accuracy']);
        
        return;


    def train_NN(self, X_input, y_output, X_val=[], y_val=[]):

        # Add a channels dimension.
        X_input =   X_input[..., tf.newaxis];

        if not X_val:
            val_data = None;

        #print(X_input.shape, y_output.shape);
        self.train_history   =   self.model.fit(     
                                            x = X_input,
                                            y = y_output, 
                                            #(X_input, y_output),
                                            epochs  =   self.epochs,
                                            validation_split = 0.15,
                                            #batch_size=BATCH_SIZE,
                                            #validation_data = val_data, 
                                            callbacks = None,                             
                                            verbose = 1);
        return self.train_history;
    
    def plot_training_model(self, history = []):

        if not history:
            history =   self.train_history;

        plt.plot(   history.history['accuracy'],        label='train_accuracy');
        plt.plot(   history.history['val_accuracy'],    label = 'val_accuracy');

        plt.xlabel('Epoch');
        plt.ylabel('Accuracy');
        plt.legend(loc='lower right');

        plt.ylim(   [0.3, 1]);
        plt.show();

        return;

    def predict_funct(self, X_input):
        X_input     =   X_input.astype(np.float64);

        # Add a channels dimension.
        X_input =   X_input[..., tf.newaxis];  

        prob_output =   self.model.predict(X_input);
        prob_output =   Softmax(prob_output)[0].numpy();

        predicted_digit =   np.argmax(prob_output);

        return  predicted_digit, prob_output;

    def test_accuracy(self, X_input, y_output):
        ## Returns the % accuracy rate of the predicted output vs. data output.

        # Add a channels dimension.
        X_input =   X_input[..., tf.newaxis];  

        eval_loss, eval_accuracy  =   self.model.evaluate(     X_input,  y_output, verbose=2);
        return eval_loss, eval_accuracy ;


def one_hot(y_input):
    ##  Converts a class vector (integers) to binary class matrix.  
    ##  Input:
    ##      y_input: np.array: 1-dim vector (numpy).
    ##      
    y_one_hot = tf.keras.utils.to_categorical(   y_input, np.max(y_input)+1);
    y_one_hot = y_one_hot.astype(np.float64);
    return y_one_hot;
