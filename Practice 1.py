#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import warnings
warnings.filterwarnings('ignore')


# In[7]:


insurance = pd.read_csv('insurance.csv')


# In[8]:


insurance.head()


# In[9]:


insurance.info()


# In[10]:


insurance.isnull().sum()


# In[11]:


insurance.describe()


# ### EXPLORATORY DATA ANALYSIS

# In[12]:


#we can apply log tranform to correst skewness
skewed = sns.displot(insurance['expenses']) #charges is right-skewed
skewed

log_trans = sns.displot(np.log10(insurance['expenses'])) #skewness is corrected using log
log_trans


# In[13]:


expenses = insurance['expenses'].groupby(insurance['region']).sum().sort_values(ascending=False)
expenses


# In[14]:


#to display grid in plot
sns.set(style="whitegrid")


# In[15]:


#ploting histogram and dist plot to see distribution of 'bmi' column
plt.hist(insurance['bmi'], color= 'green', edgecolor = 'black', alpha = 0.6)
plt.xlabel('bmi')
plt.show()

sns.distplot(insurance['bmi'], color= 'green')


# In[16]:


#ploting histogram and dist plot to see distribution of 'age' column
plt.hist(insurance['age'], color= 'blue', edgecolor = 'black', alpha = 0.6)
plt.xlabel('age')
plt.show()

sns.distplot(insurance['age'], color= 'blue')


# In[18]:


#ploting histogram and dist plot to see distribution of 'expenses' column
plt.hist(insurance['expenses'], color= 'orange', edgecolor = 'black', alpha = 0.6)
plt.xlabel('expenses')
plt.show()

sns.distplot(insurance['expenses'], color= 'orange')


# In[20]:


#WE CHECK THE EXPENSES BY REGION WHO ARE SMOKERS
sns.barplot(x='region', y='expenses', data=insurance, hue='smoker', palette='coolwarm')


# In[21]:


#WE CHECK THE EXPENSES BY REGION BY THIER SEX
sns.barplot(x='region', y='expenses', data=insurance, hue='sex', palette='rocket')


# In[22]:


#WE CHECK THE EXPENSES BY REGION BY THE CHILDREN AVAILABLE
plt.figure(figsize=(12,6))
sns.barplot(x='region', y='expenses', data=insurance, hue='children', palette='Paired')


# In[25]:


plt.figure(figsize=(12,6))
sns.violinplot(x='children', y='expenses', data=insurance, hue='smoker',split=False, palette='rocket')


# In[26]:


#regression plot to understand the relationship between the bmi and expenses considering
sns.lmplot(x="bmi", y="expenses", row="sex", col="region", hue='smoker', data=insurance)


# In[27]:


#regression plot to understand the relationship between the age and expenses considering
sns.lmplot(x="age", y="expenses", row="sex", col="region", hue='smoker', data=insurance)


# In[28]:


#WE CONVERT OBJECT LABELS INTO CATEGORICAL DATA TYPE
insurance[['sex','region','smoker']] = insurance[['sex','region','smoker']].astype('category')
insurance.dtypes


# In[29]:


##Converting category labels into numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(insurance.sex.drop_duplicates())
insurance.sex = label.transform(insurance.sex)
label.fit(insurance.smoker.drop_duplicates())
insurance.smoker = label.transform(insurance.smoker)
label.fit(insurance.region.drop_duplicates())
insurance.region = label.transform(insurance.region)


# In[30]:


insurance.head()


# In[31]:


plt.figure(figsize=(10,8))
sns.heatmap(insurance.corr(),cmap='coolwarm',annot=True)


# ### WE SPLIT THE DATA INTO PREDICTOR AND RESPONSE VARIABLE

# In[32]:


#we split our model
X = insurance.drop(['expenses'], axis = 1)
y = insurance['expenses']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# ## MODEL BUILDING

# In[36]:


import statsmodels.api as sm #WE GET THE STATISTICAL MODEL
#add constant to predictor variables
X2 = sm.add_constant(X_train)
#fit linear regression model
model = sm.OLS(y_train, X2).fit()


# In[37]:


model.summary()


# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# R-squared: 0.747. This is known as the coefficient of determination. It is the proportion of the variance in the
# response variable that can be explained by the predictor variables. In this example, 74.7% of the variation in
# the exam scores can be explained by the number of hours studied and the number of prep exams taken.
# 
# F-statistic: 523.6. This is the overall F-statistic for the regression model.
# 
# Prob (F-statistic): 3.16e-313. This is the p-value associated with the overall F-statistic. It tells us whether or
# not the regression model as a whole is statistically significant. In other words, it tells us if the predictor
# variables combined have a statistically significant association with the response variable. In this case the pvalue
# is less than 0.05, which indicates that the predictor variables combined have a statistically significant
# association with the response variable.
# 
# coef: The coefficients for each predictor variable tell us the average expected change in the response
# variable, assuming the other predictor variable remains constant.
# model.summary()
# 
# P>|t|. The individual p-values tell us whether or not each predictor variable is statistically significant. it
# statistically significant at α < 0.05 and not statistically significant at α => 0.05. Since “sex” is not statistically
# significant, we may end up deciding to remove it from the model.

# ## MULTIPLE LINEAR REGRESSION MODEL

# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


lr = LinearRegression()


# In[40]:


lr.fit(X_train,y_train)


# In[41]:


print('Intercept', lr.intercept_)
print('Coefficient', lr.coef_)
print('Score', lr.score(X_test, y_test))


# In[42]:


coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[44]:


#WE PREDICT OUR MODEL
y_pred1 = lr.predict(X_test)


# In[45]:


plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred1, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Charges')
plt.xlabel('Charges')
plt.ylabel('')
plt.show()
plt.close()


# ANOTHER PLOTTING TECHNIQUE

# In[46]:


plt.figure(figsize=(12, 8))
# acutal values
plt.plot([i for i in range(len(y_test))],np.array(y_test), c='g', label="actual values")
# predicted values
plt.plot([i for i in range(len(y_test))],y_pred1, c='m',label="predicted values")
plt.legend()


# In[47]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
predicted1.head()


# ### LASSO REGRESSION MODEL

# In[48]:


from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold


# In[49]:


#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=4, random_state=101) #we’ll use the RepeatedKFold
#define model
lasso_model = LassoCV(alphas=(0.1, 1.0, 10.0), cv=cv, n_jobs=-1)
#fit model
lasso_model.fit(X_train, y_train)


# In[50]:


print(lasso_model.intercept_)
print(lasso_model.coef_)
print(lasso_model.score(X_test, y_test))


# In[51]:


#WE PREDICT OUR LASSOCV REGRESSION MODEL
y_pred2 = lasso_model.predict(X_test)


# In[53]:


# Visualising the Lasso Regression results
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred2, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Expenses')
plt.xlabel('Expenses')
plt.ylabel('')
plt.show()
plt.close()


# In[54]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
predicted2.head()


# ## RIDGE REGRESSION MODEL

# In[55]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold


# In[57]:


#define cross-validation method to evaluate model
rid = RepeatedKFold(n_splits=10, n_repeats=3, random_state=101)


# In[63]:


#define model
rid_model = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv, scoring='neg_mean_absolute_error')
#fit model
rid_model.fit(X_train, y_train)


# In[65]:


print(rid_model.intercept_)
print(rid_model.coef_)
print(rid_model.score(X_test, y_test))


# In[66]:


#WE PREDICT OUR RIDGECV REGRESSION MODEL
y_pred3 = rid_model.predict(X_test)


# In[67]:


# Visualising the Ridge Regression results
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred3, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Expenses')
plt.xlabel('Expenses')
plt.ylabel('')
plt.show()
plt.close()


# In[68]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})
predicted3.head()


# ### ELASTICNET REGRESSOR

# In[69]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet


# In[70]:


#define cross-validation method to evaluate model
rid = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[72]:


# define model
net_model = ElasticNet(alpha=0.1, l1_ratio=0.9, fit_intercept=True, max_iter=1000, random_state=1)
# evaluate model
scores = cross_val_score(net_model, X_train, y_train, scoring='neg_mean_absolute_error')
net_model.fit(X_train, y_train)


# In[73]:


print(net_model.intercept_)
print(net_model.coef_)
print(net_model.score(X_test, y_test))


# In[74]:


y_pred4 = net_model.predict(X_test)


# In[75]:


# Visualising the ElasticNet Regressor results
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred4, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Expenses')
plt.xlabel('Expenses')
plt.ylabel('')
plt.show()
plt.close()


# In[76]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})
predicted4.head()


# ### RANDOM FOREST REGRESSOR

# In[107]:


from sklearn.ensemble import RandomForestRegressor


# In[129]:


rfg = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_split=2, random_state=123)

#fit the regressor model
rfg.fit(X_train, y_train)


# In[130]:


print(rfg.score(X_test, y_test))


# In[131]:


y_pred5 = rfg.predict(X_test)


# In[132]:


# Visualising the Random Forest Regressor results
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred5, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Expenses')
plt.xlabel('Expenses')
plt.ylabel('')
plt.show()
plt.close()


# In[133]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted5 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred5})
predicted5.head()


# ### WE TRY SELECT THE BEST FEATURES USING FEATURE IMPORTANCE FROM RANDOM FOREST REGRESSOR

# In[134]:


features = X.columns
importances = rfg.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# We can see that the smoker, bmi and age are more important features compared to the other features.

# In[135]:


#We select the import features
X = insurance.drop(['expenses', 'region', 'sex'], axis = 1)
y = insurance['expenses']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[136]:


X.shape


# ### WE BUILD A MODEL USING THE POLYNOMIAL REGRESSION AFTER FEATURE IMPORTANCE

# ### POLYNOMIAL REGRESSION MODEL

# In[137]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_poly, y, test_size = 0.2, random_state=1)


# In[138]:


pol_reg = LinearRegression()
pol_reg.fit(X_train, y_train)


# In[139]:


print(pol_reg.intercept_)
print(pol_reg.coef_)
print(pol_reg.score(X_test, y_test))


# In[140]:


#WE PREDICT OUR POLYNOMIAL REGRESSION MODEL
y_pred6 = pol_reg.predict(X_test)


# In[141]:


# Visualising the Polynomial Regression results
plt.figure(figsize=(12, 10))
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred6, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Expenses')
plt.xlabel('Expenses')

plt.show()
plt.close()


# In[142]:


#DATAFRAME FOR ACTUAL AND PREDICTED VALUE
predicted6 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred6})
predicted6.head()


# ### MODEL EVALUATION

# In[144]:


from sklearn import metrics
from sklearn.metrics import r2_score


# MULTIPLE LINEAR REGRESSION

# In[145]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# LASSOCV REGRESSION

# In[146]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred2))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))


# RIDGECV REGRESSION

# In[147]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred3))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))


# ELASTICNET REGRESSION

# In[148]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred4))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred4))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))


# RANDOM FOREST REGRESSOR

# In[150]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred5))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred5))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred5)))


# POLYNOMIAL REGRESSION

# In[151]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred6))
print('Mean Square Error:', metrics.mean_squared_error(y_test, y_pred6))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred6)))


# From the model evaluation score, we can see that polynomial regression and the Random Forest Regressor are performing well than the other models.
# We can conclude that smoking have an high impact on the cost of medical insurance follwed by bmi and age. 
# Sex is not really a determining factor
