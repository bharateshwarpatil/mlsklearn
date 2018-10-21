#!/usr/bin/env python
# coding: utf-8

# In[611]:


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[612]:


from sklearn.model_selection import train_test_split


# In[613]:


data = pd.read_csv('/Users/Bharat/Documents/Documents/machinelearningProjects/KaggleCompetition/Titanic/all/train.csv')
data #print the data set for the decision Tree


# In[614]:


data.hist(bins=50, figsize=(20,15))


# In[615]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['Sex'])
le.classes_
data['Sex'] = le.fit_transform(data['Sex'])


# In[616]:


Survived =data['Survived']
passengerIds=data['PassengerId']
data = data.drop(['Ticket','Name','Cabin','Embarked','Survived','Parch','SibSp','PassengerId'], axis = 1)
values = {'Age': 30}
data=data.fillna(value=values)
dataOuput=data
dataOuput


# In[617]:


data_train, data_test, Survived_train, Survived_test = train_test_split(data, Survived, test_size = 0
                                                                        .2, random_state = 42)


# In[618]:


depth = []
for i in range(2, 40):
    clf = tree.DecisionTreeClassifier(max_depth = i) # create decision 
    scores = cross_val_score(estimator = clf, X = data_train, y = Survived_train, cv = 9, n_jobs = 8)
    depth.append((i, scores.mean()))


# In[619]:


depth


# In[620]:


clf = tree.DecisionTreeClassifier(max_depth = 9)
clf.fit(data_train, Survived_train)
prediction = clf.predict(data_test)
data_test


# In[621]:


test=np.array([[1,1,7.2833,38],[3,0,78.85,80]])# 1,3 is class ,0 is for sex 1 male 0 female ,236.55 is fare 
clf.predict(test)


# In[622]:


dataOutput['Sex'] = le.fit_transform(dataOutput['Sex'])
values = {'Age': 30}
dataOutput=dataOutput.fillna(value=values)
dataOutput


# In[623]:


output=clf.predict(dataOutput)
output=np.column_stack((passengerIds,output))
output
names = np.array(['PassengerId','Survived'])
names
df = pd.DataFrame(output,columns=names)
df


# In[ ]:




