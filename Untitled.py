#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df["seller_type"].unique())
print(df["transmission"].unique())
print(df["owner"].unique())


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_dataset = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[10]:


final_dataset.head()


# In[11]:


final_dataset['current_year'] = 2024


# In[12]:


final_dataset.head()


# In[13]:


final_dataset["no_year"] = final_dataset["current_year"] - final_dataset["year"]


# In[14]:


final_dataset.head()


# In[15]:


final_dataset.drop(["year"], axis=1, inplace=True)


# In[16]:


final_dataset.head()


# In[17]:


final_dataset.drop(["current_year"], axis=1, inplace=True)


# In[18]:


final_dataset.head()


# In[19]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[20]:


final_dataset.head()


# In[21]:


final_dataset.corr()


# In[22]:


import seaborn as sns


# In[23]:


sns.pairplot(final_dataset)


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))
g = sns.heatmap(final_dataset[top_corr_features].corr(), annot = True)


# In[26]:


final_dataset.head()


# In[27]:


x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[28]:


x.head()


# In[29]:


y.head()


# In[33]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x, y)


# In[34]:


print(model.feature_importances_)


# In[42]:


feat_imp = pd.Series(model.feature_importances_, index = x.columns)
feat_imp.nlargest(5).plot(kind = "barh")
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[44]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()


# In[47]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[48]:


from sklearn.model_selection import RandomizedSearchCV


# In[49]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[50]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[51]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[52]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[54]:


rf_random.fit(x_train,y_train)


# In[55]:


rf_random.best_params_


# In[56]:


rf_random.best_score_


# In[58]:


predictions=rf_random.predict(x_test)


# In[59]:


sns.distplot(y_test-predictions)


# In[60]:


plt.scatter(y_test,predictions)


# In[61]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[62]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

