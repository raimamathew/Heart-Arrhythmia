#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


get_ipython().system('pip show tensorflow')


# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


get_ipython().system('unzip -o mitbih_train.zip')


# In[88]:


train_data =  pd.read_csv('mitbih_train.csv')
test_data = pd.read_csv('mitbih_test.csv')
train_data.shape


# In[6]:


M = train_data.values
X = M[:, :-1]
y = M[:, -1]

C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()


# In[7]:



# example of ECG signal for each class
x = np.arange(0, 187)*8/1000

plt.figure(figsize=(20,12))
plt.plot(x, X[C0, :][0], label="Class 0")
plt.plot(x, X[C1, :][0], label="Class 1")
plt.plot(x, X[C2, :][0], label="Class 2")
plt.plot(x, X[C3, :][0], label="Class 3")
plt.plot(x, X[C4, :][0], label="Class 4")
plt.legend()
plt.title("ECG signal for each category", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
plt.show()


# In[8]:


# feature engineering
def stretch(x):
    l = int(187 * (1 + (random.random()-0.5)/3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x):
    result = np.zeros(shape= (4, 187))
    for i in range(3):
        if random.random() < 0.33:
            new_y = stretch(x)
        elif random.random() < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        result[i, :] = new_y
    return result


# In[9]:


# plot of the Class 0 signal after being modified
import random
from scipy.signal import resample
plt.plot(X[0, :])
plt.plot(amplify(X[0, :]))
plt.plot(stretch(X[0, :]))
plt.show()


# In[56]:


# Set the train and validation dataset
result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)
classe = np.ones(shape=(result.shape[0],), dtype=int)*3
X = np.vstack([X, result])
y = np.hstack([y, classe])

subC0 = np.random.choice(C0, 400)
subC1 = np.random.choice(C1, 400)
subC2 = np.random.choice(C2, 400)
subC3 = np.random.choice(C3, 400)
subC4 = np.random.choice(C4, 400)


# In[63]:


from sklearn.utils import shuffle

XNN_val = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
yNN_val = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

XNN_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
yNN_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

XNN_train, yNN_train = shuffle(XNN_train, yNN_train, random_state=10)
XNN_val, yNN_val = shuffle(XNN_val, yNN_val, random_state=10)
print("XNN_train", XNN_train.shape)
print("yNN_train", yNN_train.shape)
print("XNN_val", XNN_val.shape)
print("yNN_val", yNN_val.shape)


# In[64]:


from sklearn.preprocessing import OneHotEncoder
# transform data for modelling
XNN_train = np.expand_dims(XNN_train, 2)
XNN_val = np.expand_dims(XNN_val, 2)
ohencoder = OneHotEncoder()
yNN_train = ohencoder.fit_transform(yNN_train.reshape(-1,1))
yNN_val = ohencoder.transform(yNN_val.reshape(-1,1))
print("XNN_train", XNN_train.shape)
print("yNN_train", yNN_train.shape)
print("XNN_val", XNN_val.shape)
print("yNN_val", yNN_val.shape)


# In[65]:


n_obs, feature, depth = XNN_train.shape
batch_size = 500


# In[66]:


from keras.layers import Input,Conv1D, Activation, Add, MaxPooling1D, Dense, Softmax, Flatten
from keras.models import Model


# In[68]:


thresh = (sum(y)/len(y))
thresh


# In[69]:


inp = Input(shape=(feature, depth))
C = Conv1D(filters=32, kernel_size=5, strides=1)(inp)

C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C)
A11 = Activation("relu")(C11)
C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A11)
S11 = Add()([C12, C])
A12 = Activation("relu")(S11)
M11 = MaxPooling1D(pool_size=5, strides=2)(A12)


C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M11)
A21 = Activation("relu")(C21)
C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A21)
S21 = Add()([C22, M11])
A22 = Activation("relu")(S11)
M21 = MaxPooling1D(pool_size=5, strides=2)(A22)


C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M21)
A31 = Activation("relu")(C31)
C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A31)
S31 = Add()([C32, M21])
A32 = Activation("relu")(S31)
M31 = MaxPooling1D(pool_size=5, strides=2)(A32)


C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M31)
A41 = Activation("relu")(C41)
C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A41)
S41 = Add()([C42, M31])
A42 = Activation("relu")(S41)
M41 = MaxPooling1D(pool_size=5, strides=2)(A42)


C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)
A51 = Activation("relu")(C51)
C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)
S51 = Add()([C52, M41])
A52 = Activation("relu")(S51)
M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

C61 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M51)
A61 = Activation("relu")(C61)
C62 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A61)
S61 = Add()([C62, M51])
A62 = Activation("relu")(S61)
M61 = MaxPooling1D(pool_size=5, strides=2)(A62)





F1 = Flatten()(M61)

D1 = Dense(32)(F1)
A6 = Activation("relu")(D1)
D2 = Dense(32)(A6)
D3 = Dense(5)(D2)
A7 = Softmax()(D3)

model = Model(inputs=inp, outputs=A7)

model.summary()


# In[19]:


from keras.callbacks import LearningRateScheduler
from sklearn.metrics import precision_score
from keras.optimizers import Adam


# In[70]:


# run the model
import math

def exp_decay(epoch):
    initial_lrate = 0.001
    k = 0.75
    t = n_obs//(10000 * batch_size)  # every epoch we do n_obs/batch_size iteration
    lrate = initial_lrate * math.exp(-k*t)
    return lrate

lrate = LearningRateScheduler(exp_decay)



adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)



model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(XNN_train, yNN_train, 
                    epochs=50, 
                    batch_size=batch_size, 
                    verbose=2, 
                    validation_data=(XNN_val, yNN_val), 
                    callbacks=[lrate])


# In[71]:


from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


# In[72]:


from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score


# In[ ]:


#Evaluation of the model begis here


# In[74]:



# check on validation set
yNN_pred = model.predict(XNN_val, batch_size=1000)
print("log loss : {:.4f}".format(log_loss(yNN_val.todense(), yNN_pred)))
print(classification_report(yNN_val.argmax(axis=1), yNN_pred.argmax(axis=1)))
# comparable result as RFC, prediction are better for the minority groups


# In[ ]:





# In[75]:


# predict test set
M = test_data.values
X_test = M[:, :-1]
y_test = M[:, -1]


# In[76]:


X_test = np.expand_dims(X_test, 2)
ohencoder = OneHotEncoder()
y_test = ohencoder.fit_transform(y_test.reshape(-1,1))

print("X_test", X_test.shape)
print("y_test", y_test.shape)


# In[ ]:





# In[77]:


from sklearn.metrics import confusion_matrix


# In[78]:


confusion_matrix = pd.DataFrame(np.array(confusion_matrix(y_test.argmax(axis=1), y_pred2.argmax(axis=1))),
                         index = ['0', '1','2', '3','4'],
                        columns=['pred_0', 'pred_1','pred_2', 'pred_3','pred_4'])
confusion_matrix


# In[79]:



y_pred2 = model.predict(X_test)
print("log loss: {:.3f}".format(log_loss(y_test.todense(), y_pred2)))
print(classification_report(y_test.argmax(axis=1), y_pred2.argmax(axis=1)))
# slightly better result than RFC for the test set


# In[ ]:




