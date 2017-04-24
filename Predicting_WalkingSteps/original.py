
# coding: utf-8

# In[59]:

import keras
import tensorflow as tf
# get_ipython().magic('matplotlib inline')
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[60]:

print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))


# In[61]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 1
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


# In[62]:

model.summary()


# In[ ]:




# In[63]:

import pandas as pd
from random import random
import numpy as np

#x = np.linspace(-np.pi, np.pi, 201)
x = np.arange(0, 2000)
data = np.sin(x/2.0) + np.cos(x/5.0)
data_noized = data * (1+random()) * 0.5
print(data)
print(data.size)

# データ作成
'''
a = np.arange(1,10,1)
b = np.arange(10,1,-1)
data = np.hstack((a, b))
data = np.tile(data, 100)
data_noized = data * (1 + random()) * 0.5

print(data_noized)
print(data_noized.size)
'''

# In[80]:

import numpy as np

def load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev])
        docY.append(data[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(data, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(data) * (1 - test_size))

    X_train, y_train = load_data(data[0:ntrn])
    X_test, y_test = load_data(data[ntrn:])

    print(X_train.shape)
    print(y_train.shape)

    X_train = X_train[:, :, np.newaxis]
    y_train = y_train[:, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    y_test = y_test[:, np.newaxis]

    print(X_train.shape)

    return (X_train, y_train), (X_test, y_test)


# In[81]:

(X_train, y_train), (X_test, y_test) = train_test_split(data_noized)  # retrieve data

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=450, epochs=100, validation_split=0.05)


# In[57]:
print("---")
print(X_test)
predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print("---")
print(predicted)
# In[58]:
print("---")
print(y_test)

import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (17, 9)
plt.plot(predicted[:100][:,0],"--")
#plt.plot(predicted[:100][:,1],"--")
plt.plot(y_test[:100][:,0],":")
#plt.plot(y_test[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()

# In[ ]:
