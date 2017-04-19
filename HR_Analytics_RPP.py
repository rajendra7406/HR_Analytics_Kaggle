
# coding: utf-8

# # Introduction

# Rajendra Prasad Patil
# 
# Analysing HR Data of left & stayed employess.
# Factors affecting employees resignations.
# 
# 1. Introduction
#     - Import Libraries
#     - Load Dataset
#     - Find Missing Values (if so fill the missing values)
#     - Preparing the Dataset
# 2. Visualizations
#     - Showing Important Features
#     - Other Visualizations
#     - Correlation with target variable
# 
# 3. Feature Engineering.
# 4. Conclusion

# # Import Libraries 
# 

# In[1]:

import numpy as np #to read the file
import pandas as pd #for numerical computations


# # Load Data

# In[2]:

# Importing the dataset using pandas library
dataset = pd.read_csv('HR_comma_sep.csv')
#prints first 5 rows
dataset.head()


# In[3]:

#Renaming of dataset
dataset=dataset.rename(columns={'sales':'dept'})
dataset=dataset.rename(columns={'average_montly_hours':'average_monthly_hours'})


# In[4]:

#Gives feature names, type, entry counts, feature count, memory usage etc
dataset.info() 


# 14999 training examples and 9 features (column "left" is result not feature)

# In[5]:

#lets see if there are any more columns with missing values 
dataset.isnull().sum()


# Luckily no missing data

# # Visualizations

# In[6]:

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=["dept","salary"]
for col in cat_vars:
    dataset[col]=labelEnc.fit_transform(dataset[col])
#showing results for less confusion
#for salary, low=1,mid=2,high=0
        


# In[7]:

#for all the plots to be in line
get_ipython().magic('matplotlib inline')
#matplot.lib for plotting 
import matplotlib.pyplot as plt
plt.style.use(style = 'default')
dataset.hist(bins=11,figsize=(10,10),grid=True)


# Visulaizing which features contribute the most using RandomForestClassifier

# In[8]:

#Assuming RandomForestClassifier is best.
from sklearn.ensemble import RandomForestClassifier

predictors = ["satisfaction_level", "last_evaluation", "number_project",
              "average_monthly_hours","time_spend_company","Work_accident", "promotion_last_5years", "dept","salary"]
rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
    
rf.fit(dataset[predictors],dataset["left"])
importances=rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
sorted_important_features=[]
for i in indices:
    sorted_important_features.append(predictors[i])
plt.figure()
plt.title("Feature Importances By Random Forest Model")
plt.bar(range(np.size(predictors)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

plt.xlim([-1, np.size(predictors)])
plt.show()


# In[9]:

#Heat Map is drawn
import seaborn as sns
sns.set(font_scale=1)
corr=dataset.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between features')


# # Further Visualizations
# Satisfaction Level, no of projects, time spent in company, average monthly hours, last evaluation are the important features.
# In future, various plots will be drawn using cartesian product of the above features.

# In[10]:

import seaborn as sns
sns.set(font_scale=1)
g = sns.FacetGrid(dataset, col="number_project", row="left", margin_titles=True)
g.map(plt.hist, "satisfaction_level",color="green")


# 1. Employees with less satisfaction for 2 & 6 projects left.
# 2. Rest most stayed in the company. 
# 3. Asking satisfaction level and their count of projects will stand alone decide employee leaving.
# Further Observations are given below.

# In[11]:

g = sns.FacetGrid(dataset, hue="left", col="time_spend_company", margin_titles=True,
                  palette={1:"black", 0:"red"})
g=g.map(plt.scatter, "satisfaction_level", "average_monthly_hours",edgecolor="w").add_legend()


# 1. Employees who stayed longer remained & unfortunately employees are less in no.
# 2. Early employees have more satisfaction and it slowly decreases with more no of years in company. 
# 3. employees are leaving after 5,6 years inspite of high satisfaction
# 4. Hard working employees are leaving during 4,5 years because of low satisfaction
# 5. Lazy emplyees tend to leave during 3rd year with medium satisfaction level.

# In[12]:

g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "average_monthly_hours", "time_spend_company",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Resignation by time spent in compnay, no of projects, average monthly hours spent')


# 1. Most of the employees worked equal hours untill 6 years until 6  projects. 
# 2. Its interesting as early employees didnt resign inspite long working hours.
# 3. 4-6 years experience with 7 projects have 99% of chance resigning.
# 4. If projects are less inspite of working 3 years and spending less time, they are gonna resign, it shows lack of commitment.
# 5. 5-6 experience employees leave leave after 4-5 jobs for better jobs with more working time.
# 5. Employees with 6 projects in 4 years leave company with more working time.

# In[13]:

g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,
                  palette={1:"brown", 0:"green"})
g=g.map(plt.scatter, "satisfaction_level", "last_evaluation",edgecolor="w").add_legend()


# 1. Employees in 3 projects arent leaving.
# 2. But employees having higher satisfaction, high latest evaluation rates leave after 4/5 projects leave
# 3. Employees with very low satisfaction after 6-7 projects inspite of good evaluation rates are more likely to leave
# 4. Employees with 2 projects, with low satisfaction level and evaluation results leave. I see in-efficiency

# In[14]:

sns.set(font_scale=1)
g = sns.factorplot(x="number_project", y="left", col="time_spend_company",
                    data=dataset, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("no of projects", "leaving Rate")
    .set_xticklabels([1,2,3,4,5,6,7])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many employees left completing projects and time spending in company')


# 1. Employees who complete kinda more projects in less years tend to leave, they are smart employees. 
# 2. Just 1 project in course of 3 years, employees is leaving, I think he might be bored.
# 3. Employees in course of 5 years in company tend to leave with more projects in search of better jobs.
# 4. Interestingly, after 7 years in company, no one wants to leave. I bet they would have become managers.

# In[15]:

g = sns.FacetGrid(dataset, hue="left", col="number_project", margin_titles=True,
                  palette={1:"yellow", 0:"orange"})
g=g.map(plt.scatter, "last_evaluation", "average_monthly_hours",edgecolor="w").add_legend()


# 1. Lazy employees for 2 projects leave the company
# 2. Again, hard working employees with 4-6 projects leave after good latest evaluation for better jobs.
# 3. 99% of hardworking employees with more projects leave the company. 

# # In a nut-shell
# 1. Employees with 4-6 projects and 4-6 years experience are more likely to leave.
# 2. Smart employees who complete more projects in less years leave.
# 3. Lazy, inefficient, bored, less satisfied employees leave.
# 4. Employees with more than 6 years of experience remain in the company.
# These are the 4 kinds of employees.

# # Feature Engineering
# Here, I have combined features to get a narrowed analysis on resignation of an employee.

# In[28]:

dataset['efficiency'] = ( dataset['time_spend_company'] * (12) * dataset['average_monthly_hours'] )/ dataset['number_project']
#12 months in a year
_ = sns.distplot(dataset['efficiency'])
plt.show()
x1 = np.corrcoef(x=dataset['efficiency'], y=dataset['satisfaction_level'])
y1 = np.corrcoef(x=dataset['efficiency'], y=dataset['left']) 
z1 = np.corrcoef(x=dataset['left'], y=dataset['satisfaction_level']) 
print(x1,y1,z1)


# Combining important features to get a better understanding. 
# As you can observe from correlation values listed above.
# The "efficiency has positive correlation on resignation". 
# This is kinda spooky to tell, as efficiency increases leaving rate too increases.
# The efficiency feature corelation value when compared to original corelation matrix, is second highest. 

# # Conclusions
# I am giving out one conclusion,
# if the company wants to retain its valuable employers, it has to reduce/increase one of the factors in the efficiency.
# If so, either they have to increase no of projects or decrease average monthly hours, which is kinda impractical if both done at same time. 

# This is the end of the notebook as for now, will update soon as my knowledge horizons increase. Ty for observing this notebook.

# Please, this is for expert data scientists, if you find any errors, leave in the comment section below, i will definitely re write the code. 
# If you like my work and want to work in team. Please ping me, because I am noobie and want to learn a lot. If there is anything I have to add, let me know in the comments section below. Thanks in advance. 
