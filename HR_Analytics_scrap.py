# Importing the libraries
import numpy as np 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HR_comma_sep.csv')
X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y = dataset.iloc[:, 6].values

#find missing data
dataset.info() 
#lets see if there are any more columns with missing values 
dataset.isnull().sum()

#visualizations
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)
plt.style.use(style = 'default')
dataset.hist(bins=11,rwidth=13, figsize=(10,7),grid=False)
#
g = sns.FacetGrid(dataset, col="dept", row="left", margin_titles=True)
g.map(plt.hist, "satisfaction_level",color="purple")
#
g = sns.FacetGrid(dataset, hue="left", col="salary", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "promotion_last_5years", "time_spend_company",edgecolor="w").add_legend()

##__
ax = sns.boxplot(x="left", y="average_monthly_hours", 
                data=dataset)
ax = sns.stripplot(x="left", y="last_evaluation",
                   data=dataset, jitter=True,
                   edgecolor="gray")
sns.plt.title("Leaving based on no of projects and time spent in company",fontsize=12)

#__
g = sns.FacetGrid(dataset, hue="left", col="dept", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "average_monthly_hours", "last_evaluation",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare')

##
dataset.number_project.value_counts().plot(kind='bar', alpha=0.55)
plt.title("workers project count")
##
sns.factorplot(x = 'number_project',y="left", data = dataset,color="r")
##
g = sns.FacetGrid(dataset, col="average_monthly_hours", row="left", margin_titles=True)
g.map(plt.hist, "last_evaluation",color="black")
#
sns.set(font_scale=1)
g = sns.factorplot(x="salary", y="left", col="Work_accident",
                    data=dataset, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "leaving Rate")
    .set_xticklabels(["low", "medium","high"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many workers stayed after injuries')
#
ax = sns.boxplot(x="left", y="satisfaction_level", 
                data=dataset)
ax = sns.stripplot(x="left", y="satisfaction_level",
                   data=dataset, jitter=True,
                   edgecolor="gray")
sns.plt.title("Leaving by satisfaction",fontsize=12)
##
"""
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

predictors = ["satisfaction_level", "last_evaluation", "number_project",
              "average_monthly_hours","time_spend_company","Work_accident", "promotion_last_5years", "dept","salary"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, 
                            min_samples_leaf=1)
kf = KFold(dataset.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)



predictions = cross_validation.cross_val_predict(rf, dataset[predictors],dataset["left"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, dataset[predictors], dataset["Survived"],
                                          scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
"""
from sklearn.ensemble import RandomForestClassifier

predictors = ["satisfaction_level", "last_evaluation", "number_project",
              "average_monthly_hours","time_spend_company","Work_accident", "promotion_last_5years", "dept","salary"]
rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
#
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=["dept","salary"]
for col in cat_vars:
    dataset[col]=labelEnc.fit_transform(dataset[col])
    
rf.fit(dataset[predictors],dataset["left"])
importances=rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
sorted_important_features=[]
for i in indices:
    sorted_important_features.append(predictors[i])
#predictors=titanic.columns
plt.figure()
plt.title("Feature Importances By Random Forest Model")
plt.bar(range(np.size(predictors)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

plt.xlim([-1, np.size(predictors)])
plt.show()
#---------------------------
#-----------------------------------------------------------------
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
dataset[:, 8] = labelencoder_X.fit_transform(dataset[:,8 ])
dataset[:, 9] = labelencoder_X.fit_transform(dataset[:,9 ])
#column 'dept' is the categorical column
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
