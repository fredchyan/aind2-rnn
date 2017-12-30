import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [] 
    y = []
    n = len(series)
    for i in range(0, n - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    # (timesteps, data_dim) = (window_size, 1)
    model.add(LSTM(units = 5, 
                      input_shape=(window_size, 1)))
    model.add(Dense(units = 1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    res = []
    punctuation = ['!', ',', '.', ':', ';', '?']
    alpha = set("abcdefghijklmnopqrstuvwxyz")
    st = alpha | set(punctuation) 
    for i in range(len(text)):
        if text[i] in st:
           res.append(text[i]) 
        else:
           res.append(" ")
    return "".join(res) 

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # number of pairs if step_size == 1, len(text) - window_size
    for ind in range(0, len(text) - window_size, step_size):
        inputs.append(text[ind:ind+window_size])
        outputs.append(text[ind+window_size])
    

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(units=num_chars, activation='softmax'))
    return model
