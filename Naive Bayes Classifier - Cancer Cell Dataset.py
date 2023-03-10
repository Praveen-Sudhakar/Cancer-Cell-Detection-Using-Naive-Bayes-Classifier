#!/usr/bin/env python
# coding: utf-8

# In[59]:


#Importing necessary packages

import numpy as np
import pandas as pd


# In[60]:


#Reading the dataset

df = pd.read_csv("D:\AIML\Dataset\Cellsamples.csv")

df


# In[61]:


df.info()


# In[62]:


#Dropping the object column

df.drop(['BareNuc'],axis=1,inplace=True)


# In[63]:


df


# In[86]:


#feat = df[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BlandChrom","NormNucl","Mit"]]


# In[101]:


x = df[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BlandChrom","NormNucl","Mit"]].values

x[0:5]


# In[109]:


y = df['Class'].values

y[0:5]


# In[110]:


#Splitting the dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

print("Train shape",x_train.shape,y_train.shape)
print("Test shape",x_test.shape,y_test.shape)


# In[111]:


#Modeling

from sklearn.naive_bayes import GaussianNB

cellgb = GaussianNB()

cellgb.fit(x_train,y_train)


# In[112]:


#Evaluating the model using test dataset

y_pred = cellgb.predict(x_test)


# In[113]:


from sklearn.metrics import f1_score

print(f"Accuracy score = {f1_score(y_pred,y_test,average='weighted')*100} %")


# In[114]:


y_pred[0:5]


# In[115]:


y_test[0:5]


# In[116]:


#plotting

import matplotlib.pyplot as plt

plt.scatter(x_test[:,1],x_test[:,-2],c=y_test)


# In[117]:


plt.scatter(x_test[:,1],x_test[:,-2],c=y_pred)


# In[ ]:




