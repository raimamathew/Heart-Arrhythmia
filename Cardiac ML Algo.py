#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


get_ipython().system('unzip -o mitbih_train.zip')


# In[5]:


train_data =  pd.read_csv('mitbih_train.csv')


# In[6]:


x_train = pd.read_csv('mitbih_train.csv',header=None,usecols=range(187))


# In[7]:


x_train.shape


# In[8]:


y_train = pd.read_csv('mitbih_train.csv',header=None,usecols=[187]).iloc[:,0]


# In[9]:


y_train.shape


# In[10]:


x_test = pd.read_csv('mitbih_test.csv',header=None,usecols=range(187))
y_test = pd.read_csv('mitbih_test.csv',header=None,usecols=[187]).iloc[:,0]


# In[11]:


x_test.shape


# In[12]:


y_test.shape


# In[13]:


from scipy.signal import gaussian, decimate
from scipy.sparse import csr_matrix


# In[14]:


def gaussian_smoothing(data, window, std):
    gauss = gaussian(window ,std, sym=True)
    data = np.convolve(gauss/gauss.sum(), data, mode='same')
    return data

def gauss_wrapper(data):
    return gaussian_smoothing(data, 12, 7)

fig = plt.figure(figsize=(8,4))
plt.plot(x_train.iloc[1,:], label="original")
plt.plot(gauss_wrapper(x_train.iloc[1,:]), label="smoothed")
plt.legend()


# In[15]:


def plot(x_data, y_data, classes=range(5), plots_per_class=10):

    f, ax = plt.subplots(5, sharex=True, sharey=True, figsize=(10,10))
    for i in classes:
        for j in range(plots_per_class):
            ax[i].set_title("class{}".format(i))
            ax[i].plot(x_data[y_data == i].iloc[j,:], color="blue", alpha=.5)
            
plot(x_train, y_train)


# In[16]:


def gradient(data, normalize=True):
    data = data.diff(axis=1, periods=3)
    if normalize:
        data = data.apply(lambda x: x/x.abs().max(), axis=1)
    return data

def preprocess(data): 
    data = data.abs().rolling(7, axis=1).max()
    data = data.fillna(method="bfill",axis=1)
    #data = np.apply_along_axis(gauss_wrapper, 1, data)
    data = decimate(data, axis=1, q=5)
    data[np.abs(data) < .05] = 0
    return pd.DataFrame(data)

x_train_grad = gradient(x_train)
x_test_grad = gradient(x_test)

x_train_preprocessed = preprocess(pd.concat([x_train, x_train_grad, gradient(x_train_grad)], axis=1))
x_test_preprocessed = preprocess(pd.concat([x_test, x_test_grad, gradient(x_test_grad)], axis=1))


# In[17]:


plot(x_train_preprocessed, y_train)


# In[18]:


x_train_sparse = csr_matrix(x_train_preprocessed)


# In[19]:


del x_train_grad
del x_test_grad


# In[20]:


import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


# In[21]:


model = LogisticRegression(multi_class="ovr",solver="newton-cg", class_weight="balanced",
                          n_jobs=2, max_iter=150, C=.5)

start_time = time.time()
model.fit(x_train_sparse,y_train)
print("training time {}".format(time.time()-start_time))


# In[22]:


y_predict = model.predict(x_test_preprocessed)
cf = confusion_matrix(y_test,y_predict)
print("accuracy: " + str(accuracy_score(y_test,y_predict)))


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss

rfc1 = RandomForestClassifier()
rfc1.fit(x_train_preprocessed, y_train)

y_pred = rfc1.predict(x_test_preprocessed)

yhat_pp = rfc1.predict_proba(x_test_preprocessed)
print('log loss:', log_loss(y_test, yhat_pp))

print(classification_report(y_test, y_pred,))


# In[ ]:





# In[25]:


get_ipython().system('pip install xgboost')


# In[26]:


# XGBoost
import xgboost as xgb


# In[27]:


model=xgb.XGBClassifier(random_state=1,learning_rate=0.1)
model.fit(x_train_preprocessed,y_train)
model.score(x_test_preprocessed,y_test)

y_pred = model.predict(x_test_preprocessed)

yhat_pp = model.predict_proba(x_test_preprocessed)
print('log loss:', log_loss(y_test, yhat_pp))
print(classification_report(y_test,y_pred))
# Slightly worse f1-score, but better log loss score than RFC


# In[28]:


M = train_data.values
X = M[:, :-1]
y = M[:, -1]

C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()


# In[29]:



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


# In[30]:


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


# In[31]:


# plot of the Class 0 signal after being modified
import random
from scipy.signal import resample
plt.plot(X[0, :])
plt.plot(amplify(X[0, :]))
plt.plot(stretch(X[0, :]))
plt.show()


# In[32]:


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


# In[33]:


from sklearn.utils import shuffle

XNN_val = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
yNN_val = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

XNN_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
yNN_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

XNN_train, yNN_train = shuffle(XNN_train, yNN_train, random_state=0)
XNN_val, yNN_val = shuffle(XNN_val, yNN_val, random_state=0)
print("XNN_train", XNN_train.shape)
print("yNN_train", yNN_train.shape)
print("XNN_val", XNN_val.shape)
print("yNN_val", yNN_val.shape)


# In[36]:


from sklearn.preprocessing import OneHotEncoder
# transform data for modelling
XNN_train = np.expand_dims(XNN_train, 2)
XNN_val = np.expand_dims(XNN_val, 2)
ohe = OneHotEncoder()
yNN_train = ohe.fit_transform(yNN_train.reshape(-1,1))
yNN_val = ohe.transform(yNN_val.reshape(-1,1))
print("XNN_train", XNN_train.shape)
print("yNN_train", yNN_train.shape)
print("XNN_val", XNN_val.shape)
print("yNN_val", yNN_val.shape)


# In[37]:


n_obs, feature, depth = XNN_train.shape
batch_size = 500


# In[38]:


get_ipython().system('pip install tensorflow')


# In[39]:


get_ipython().system('pip install --upgrade pip')


# In[43]:


get_ipython().system('pip install keras')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




