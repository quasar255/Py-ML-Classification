#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Predict cellphone customers service level based on demographics
# 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing                     # Perprocessing
from sklearn.model_selection import train_test_split  # Split data test train
from sklearn.neighbors import KNeighborsClassifier    # Do classification training
from sklearn import metrics                           # Evaluate model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


# Import and explore/describe data
get_ipython().system('wget -O teleCust1000t.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv')
df = pd.read_csv('teleCust1000t.csv')
df.head()


# In[19]:


# check current customer classification 
print(df['custcat'].count())
df['custcat'].value_counts()


# In[20]:


df.columns


# In[21]:


# Define independent variables(Features)
# Using scikit-learn library--Convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values 
X[0:5]


# In[22]:


# Define dependent variable(label)
y = df['custcat'].values
y[0:5]


# In[25]:


# Standardize data. ZERO mean and Unit variance. preprocessing from sklearn
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[28]:


# Train test split. Again use sklearn
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[30]:


## CLASSIFICATION begins ##
k = 4

# Train Model  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predict
yhat = neigh.predict(X_test)
yhat[0:5]


# In[33]:


# Evaluate model accuracy

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[44]:


## Multiple Ks and their accuracy
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
mean_acc


# In[45]:


# Plot Ks vs accuracy
plt.plot(range(1,Ks),mean_acc,'g')
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.show()


# In[ ]:




