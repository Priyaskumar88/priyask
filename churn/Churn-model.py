#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle  # To save the model that we are going to train so that we can use it directly in our web app.


# In[2]:


data = pd.read_excel('.venv\churn - 4\E Commerce Dataset.xlsx')
data


# In[3]:


df=data


# # Exploratory Data Analysis 

# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# # Data prepocessing 

# ### Missing values filling

# In[8]:


df.isna().sum()


# In[9]:


df.info()


# In[10]:


for i in df.columns:
    if df[i].isnull().sum() > 0:
        df[i].fillna(df[i].median(),inplace=True)


# In[11]:


df.PreferredLoginDevice.replace('Mobile Phone','Phone',inplace = True)
df.PreferredPaymentMode .replace(['Credit Card','Cash on Delivery'],['CC','COD'],inplace = True)
df.PreferedOrderCat.replace('Mobile Phone','Mobile',inplace = True)


# ## one hot encoding

# In[12]:


df_encoded=df.copy()


# In[13]:


df_encoded = df_encoded.drop('CustomerID',axis=1) 
df_encoded


# In[14]:


df_encoded.columns


# In[15]:


df_encoded1=df_encoded[['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']]
df_encoded1


# In[16]:


df_encoded1 = pd.get_dummies(df_encoded1,drop_first=True)
df_encoded1


# In[17]:


df_encoded2=df_encoded.drop(['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus'],axis=1)
df_encoded2


# In[18]:


df_encoded=pd.concat([df_encoded2,df_encoded1], axis=1)
df_encoded


# In[19]:


df_encoded.columns


# In[20]:


X1 = df_encoded.drop('Churn', axis=1) 
y1 = df_encoded.Churn  


# In[21]:


y1.value_counts()


# # Handeling inbalance dataset
# 
# In this dataset we will only be handling imbalance of the Churn column which is our Target column
# 
# SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.

# In[22]:


print('Before OverSampling, the shape of X: {}'.format(X1.shape)) 
print('Before OverSampling, the shape of y: {} \n'.format(y1.shape)) 


# In[23]:


print("Before OverSampling, counts of label '1': {}".format(sum(y1 == 1))) 
print("Before OverSampling, counts of label '0': {}".format(sum(y1 == 0)))


# We can say that this is an imbalance column as ratio of 70:1 for the majority to the minority class is not satisfied

# In[24]:


from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X1, y1.ravel())


# In[25]:


print('After OverSampling, the shape of X: {}'.format(X_res.shape)) 
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape)) 


# In[26]:


print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))


# In[27]:


X_res


# In[28]:


X_res.columns


# In[29]:


X_res.info()


# In[30]:


X_res=pd.DataFrame(X_res)
#Renaming column name of Target variable
y_res=pd.DataFrame(y_res)
y_res.columns = ['Churn']
final= pd.concat([X_res,y_res], axis=1)


# In[31]:


final


# In[32]:


final.columns


# In[33]:


final.rename(columns = {'PreferredPaymentMode_Debit Card':'PreferredPaymentMode_DebitCard', 
                        'PreferredPaymentMode_E wallet':'PreferredPaymentMode_Ewallet',
                        'PreferedOrderCat_Laptop & Accessory':'PreferedOrderCat_Laptop'}, inplace = True)


# In[34]:


final.columns


# In[35]:


final


# ## train and test 

# In[36]:


# Splitting the dataset into Training and Testing Data
from sklearn.model_selection import train_test_split
X = final.drop(['Churn'],axis=1)
y = final['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=42)


# In[40]:


# Standardizing the Dataset
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler(feature_range=(0,1))
X_train =minmax.fit_transform(X_train)
X_test =minmax.transform(X_test)


# # Models

# ### RandomForestClassifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier(random_state=50,max_features=0.1, max_samples=1.0 ,n_estimators=95)


# In[42]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([('minmax', minmax), ('model', model)])


# In[43]:


model.fit(X_train,y_train)


# In[44]:


prediction = model.predict(X_test)


# In[45]:


print( classification_report(y_test, prediction) )
print(accuracy_score(y_test, prediction)*100 ,'%')


# In[46]:


RFA_pred_df = pd.DataFrame(pd.Series(prediction).value_counts(), columns=['Test Outcome'])
pd.concat([y_test.value_counts(),RFA_pred_df], axis=1)


# In[47]:


# checking training accuracy
print("training accuracy is : ", model.score(X_train, y_train)*100)


# In[48]:


# checking accuracy of test dataset
print("testing accuracy is : ", model.score(X_test, y_test)*100)


# # pickling the Model
# 

# In[49]:


# Save the trained model and scaler using pickle
pickle.dump(model,open('model.pkl','wb')) 
pickle.dump(minmax,open('minmax.pkl','wb')) 


# In[50]:


p=model.predict([[4,3,6,3,3,2,9,1,11,1,1,5,160,1,0,1,0,0,0,0,1,0,0,0,1]])
print(p)


# In[51]:


p=model.predict([[30,1,15,3,4,4,5,1,20,1,1,0,133,1,0,0,0,0,0,0,0,1,0,0,0]])
print(p)


# In[52]:


p=model.predict([[0,3,15,2,4,5,8,0,23,0,1,3,134,1,0,1,0,0,1,0,1,0,0,0,1]])
print(p)


# In[ ]:





# In[ ]:




