#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score


# In[130]:


from sklearn.model_selection import train_test_split
data = pd.read_csv('/Users/Bharat/Documents/Documents/machinelearningProjects/Artificial-Intellligence-master/iris.csv')
data.head()#print the data set for the decision Tree


# In[129]:


data.head() #just Drop Index column


# In[106]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['Species'])
le.classes_


# In[107]:


Y = le.transform(data['Species']) #numpy array
X = data.drop(['Species'], axis = 1)


# In[108]:


X.head() #print the data for the array droped the Species


# In[109]:


Y #print the species which is encoded


# In[124]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)# get the x data train data
#0.3 is 30% of the data get for the test and 70 % for the training the data ,random test will give the same set of dataset across the whole dataset now random_set 42


# In[126]:


X_test.size
X_train.size


# In[132]:


depth = []
for i in range(2, 20):
    clf = tree.DecisionTreeClassifier(max_depth = i) # create decision 
    scores = cross_val_score(estimator = clf, X = X_train, y = Y_train, cv = 5, n_jobs = 4)
    depth.append((i, scores.mean()))


# In[123]:


depth 


# In[131]:


clf = tree.DecisionTreeClassifier(max_depth = 12)
clf.fit(X, Y)
prediction = clf.predict(X_test)
le.inverse_transform(clf.predict(X_test))


# In[118]:


clf.fit(X, Y)
test=np.array([[6.1,2.8,4.7,1.2],[1.1,3.8,2.7,0.2]]) # Prdection Test #
le.inverse_transform(clf.predict(test))


# In[ ]:




