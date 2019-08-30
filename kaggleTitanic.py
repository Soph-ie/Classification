# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:40:43 2019

@author: Sophia Chisiya
"""

# importing the relevant libraries
#dependanicies to see the charts.. because %matplotlib inline is only used in a shell cmd due to '%'
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#the Python imports
import math, time, random, datetime

#imports for data manipulation
import numpy as np
import pandas as pd

#imports for visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use ('seaborn-whitegrid')

#imports for pre-processing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

#imports for machine learning .. later a bit

#ignore warnings for a bit
import warnings
warnings.filterwarnings('ignore')

#loading the data
train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
gender_submission= pd.read_csv('gender_submission.csv')

#to view the data and see whether we have uuploaded correctly
firstHead=train.head(8)
stats=train.describe()

print (stats)
train.Age.plot.hist()


#let's find out where the missing data is at
#viewing it through a plot

missingno.matrix(train, figsize=(20,20))

#we could also view missing data by getting their sum
train.isnull().sum()

#or if you want some challenge.. you could write your own function teehee
#def findMissingValues(df, columns):
#    missing_value={}
#    print('the number of missing values or NaN is:')
#    df_length= len(df)
#    for column in columns:
#        total_column_values= df[column].value_counts().sum()
#        missing_value[column]= df_length - total_column_values
#    return missing_value
#
#missing_values= findMissingValues(train, train.columns)
#missing_values


#creating new daaframes for continuous variables and discrete -continuous variables
df_bin =pd.DataFrame() #for discretized continuous variables  e.g 1-100 binned into 1, 101-200 binned into 2, 201-300 binned into 3
df_con =pd.DataFrame() #for continuous variables

#find out what kind of data is in the dataframe
# objects= categorical features
# floats/int = numerical features
train.dtypes

#exploring it individually is good

#haiya.. so how many people survived? visualize it 

fig = plt.figure(figsize= (20,1))
sns.countplot(y='Survived', data=train);
print(train.Survived.value_counts())

#let us add this to our subset dataframes
df_bin['Survived']= train['Survived']
df_con['Survived']= train['Survived']

#check it out
df_bin. head(5)
df_con. head(5)


#CHECKING OUT THE PASSEGER CLASSES
#Plot the distribution to understand the kind of spread in the dataset.....apparently if outliers are present, we do not want to plot those

sns.distplot(train.Pclass)
#how many missing values des Pclass have?
train.Pclass.isnull().sum()

#sinnce there re no mising values, we add it to our sub-dataframes
df_bin['Pclass'] =train['Pclass']
df_con['Pclass'] =train['Pclass']

    #NAME
#How many different names are there?
train.Name.value_counts()
#as there are various unique instances of names,like featureID we will not use the feature name


            #SEX
#Visualize the counts using seaborn
fig = plt.figure(figsize= (20,1))
sns.countplot(y='Sex', data=train);
print(train.Sex.value_counts())

train.Sex.isnull().sum()
#adding to the sub-dataframes
df_bin['Sex']= train['Sex']
df_bin['Sex']= np.where(df_bin['Sex']== 'female', 1,0) #change sex to 0 for male and 1 for female

df_con['Sex']= train['Sex']

#since both survived and sex are binaries; either 0 or 1, we can visually compare the two

fig=plt.figure(figsize=(10,10))
sns.distplot(df_bin.loc[df_bin['Survived']==1]['Sex'],kde_kws={'label': 'Survived'});
sns.distplot(df_bin.loc[df_bin['Survived']==0]['Sex'],kde_kws={'label': 'Did not survive'});


            #AGE
train.Age.isnull().sum()
#add more lines of code once you have decided how to deal with the missing data

#function to create count and distribution visualisations
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
                     
                            #SIBSP
train.SibSp.isnull().sum()
#what vales are there?
train.SibSp.value_counts()

# Add SibSp to subset dataframes
df_bin['SibSp'] = train['SibSp']
df_con['SibSp'] = train['SibSp']

# Visualise the counts of SibSp and the distribution of the values
# against Survived
plot_count_dist(train, 
                bin_df=df_bin, 
                label_column='Survived', 
                target_column='SibSp', 
                figsize=(20, 10))


                            #Parch
train.Parch.isnull().sum()
#what vales are there?
train.Parch.value_counts()

# Add SibSp to subset dataframes
df_bin['Parch'] = train['Parch']
df_con['Parch'] = train['Parch']

# Visualise the counts of SibSp and the distribution of the values
# against Survived
plot_count_dist(train, 
                bin_df=df_bin, 
                label_column='Survived', 
                target_column='Parch', 
                figsize=(20, 10))


#TICKET
train.Ticket.isnull().sum()
#how many kinds of tickets are there?
train.Ticket.value_counts()[:20]
sns.countplot(y='Ticket', data=train); #visually

print ('There are {} unique tickets'.format(len(train.Ticket.unique())) )

#FARE
train.Fare.isnull().sum()
train.Fare.value_counts()
print ('There are {} unique tickets'.format(len(train.Fare.unique())) )
train.Fare.dtype
#because it is a floot, we enter it into our cont dataframe but cut it into bins before we add it to our categorical dataframe
df_con['Fare']= train['Fare']
df_bin['Fare']=pd.cut(train['Fare'], bins=5)

#Visualize the Fare bin counts as well as the Fare Distribution versus Survived
plot_count_dist(data=train, 
                bin_df=df_bin, 
                label_column='Survived', 
                target_column='Fare', 
                figsize=(20, 10),
                use_bin_df=True)

# FEATURE CABIN
train.Cabin.isnull().sum()
train.Cabin.value_counts()
#toommany types and too many missing values.. ignoring it for now

#FEATURE EMBARKED
train.Embarked.isnull().sum()
train.Embarked.value_counts()
#visualizing
sns.countplot(y='Embarked', data=train)

#add the data to the dataframes

df_con['Embarked']= train['Embarked']
df_bin['Embarked']= train['Embarked']

#removing the two rows of missing data
print(len(df_con))
df_con= df_con.dropna(subset= ['Embarked'])
print(len(df_con))
df_bin= df_con.dropna(subset= ['Embarked'])

train.head()
df_bin.head()
df_con.head()

#FEATURE ENCODING




