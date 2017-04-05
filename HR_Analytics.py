#Predicting the next employee who may leave the company
# Multiple Linear Regression 
# Importing the libraries
import numpy as np 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HR_comma_sep.csv')
X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y = dataset.iloc[:, 6].values

#find missing data
dataset.info() 

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:,7 ])
X[:, 8] = labelencoder_X.fit_transform(X[:,8 ])
#column 'sales' is the categorical column
#OneHotEncoder is not applied to column 'salary' as they have relation between i.e(low>mid>high)
onehotencoder = OneHotEncoder(categorical_features=[7])
X = onehotencoder.fit_transform(X).toarray()

#now X.shape gives (14999,18)
[rowX,colX] = X.shape
"""
X_yes dataset of left employees
X_no dataset of stayed employees
The idea here is to created datasets for left employees and stayed employees
This way we can apply Backward Elimination to above 2 datasets and 
determine which independent variable influences dataset
This way, I can determine which independent variable will make employees leave
and which independent variable will make employees stay
"""
X_yes = np.zeros( (rowX,colX) )
X_no = np.zeros( (rowX,colX) )
i = 0   #iteraration variable
j = 0   #gives length of X_yes after loop
k = 0   #gives length of X_no after loop
temp=len(X)

while (temp) :
    if y[i] : 
        X_yes[j,:] = X[ i,: ].reshape(1,colX) 
        temp = temp - 1
        i = i + 1
        j = j + 1
    else :
        X_no[k,:] = X[ i,: ].reshape(1,colX) 
        temp = temp - 1
        i = i + 1
        k = k+1

#removing extra rows containing zeros
X_yes = X_yes[:-(rowX-j),: ]
X_no = X_no[:-(rowX-k),:]
 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_yes = sc_X.fit_transform(X_yes)
X_no = sc_X.transform(X_no)

#building the optimal model using backward elimination
import statsmodels.formula.api as sm
#appendding X(i) as ones to X matrix
X = np.append(arr = np.ones((len(X_yes), 1)).astype(int), values = X_yes, axis = 1)
#PL = 0.05
X_opt = X_yes[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog = np.ones((j,1)), exog = X_opt).fit()
regressor_OLS.summary()

#unfortunately
#P values are almost same. That means linear model doesnt adapt to it.
#Implement Non Linear Models such as Kernal-SVM, Random Forest, XGBoost
#Suggested ones are Kernal-SVM or XGBoost
