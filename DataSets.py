#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

def MinMax(ndarray):
    min = ndarray.min()
    max = ndarray.max()
    return min, max

#1
#Iris Data Set
def IrisDataSet():
    from sklearn.datasets import load_iris
    iris = load_iris()
    #print(iris.DESCR)
    
    x_data = iris.data
    y_target = iris.target
    attributes_names = iris.feature_names
    class_names = iris.target_names
    
    return x_data, y_target, attributes_names, class_names, iris

#2 BHP Quant
#Boston Housing Prices Data Set with Classes for five quantiles of the target regression
def BostonHPDataSetQuant():
    from sklearn.datasets import load_boston
    boston = load_boston()
    #print(boston.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = boston.data
    y_target = boston.target
    attributes_names = boston.feature_names
    class_names = ["very low", "low", "mid", "high", "very high"]
    split = len(class_names)
    
    #put x and y data together in a data frame to sort them and address a class (because regression problems have no class)
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)
    #sort by last column (target column)
    df.sort_values(by=[df.columns[-2]],inplace = True)
    #get the might of the split, at which rows the split happens, and which are the values of these split_points
    split_might = int(np.ceil(df.shape[0] / split))
    split_points = range(0  , df.shape[0] , split_might)
    split_point_values = df['target'][::split_might]

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1

    #resort data after the original index (so we just have the old data again with "new" classes)
    df.sort_index(inplace = True)
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, boston

#3 BHP Absolut
#Boston Housing Prices Data Set with Classes for five subdivisions of the absolut values of the target regression
def BostonHPDataSetAbsolut():
    from sklearn.datasets import load_boston
    boston = load_boston()
    #print(boston.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = boston.data
    y_target = boston.target
    attributes_names = boston.feature_names
    class_names = ["very low", "low", "mid", "high", "very high"]
    split = len(class_names)
    
    #put x and y data together in a data frame 
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)

    #get the might of the split and which are the values of these split_points
    target_min, target_max = MinMax(df['target'])
    split_might = (target_max - target_min) / split
    split_point_values = pd.Series(np.linspace(target_min,target_max,split,endpoint=False))

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1
    
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, boston

#4 Dia Quant
#Diabetes Data Set with Classes for three quantiles of the target regression
def DiabetesDataSetQuant():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    #print(diabetes.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = diabetes.data
    y_target = diabetes.target
    attributes_names = diabetes.feature_names
    class_names = ["bad", "same", "good"]
    split = len(class_names)
    
    #put x and y data together in a data frame to sort them and address a class (because regression problems have no class)
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)
    #sort by last column (target column)
    df.sort_values(by=[df.columns[-2]],inplace = True)
    #get the might of the split, at which rows the split happens, and which are the values of these split_points
    split_might = int(np.ceil(df.shape[0] / split))
    split_points = range(0  , df.shape[0] , split_might)
    split_point_values = df['target'][::split_might]

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1

    #resort data after the original index (so we just have the old data again with "new" classes)
    df.sort_index(inplace = True)
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, diabetes

#5 Dia Abolut
#Diabetes Data Set with Classes for three subdivisions of the absolut values of the target regression
def DiabetesDataSetAbsolut():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    #print(diabetes.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = diabetes.data
    y_target = diabetes.target
    attributes_names = diabetes.feature_names
    class_names = ["bad", "same", "good"]
    split = len(class_names)
    
    #put x and y data together in a data frame 
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)

    #get the might of the split and which are the values of these split_points
    target_min, target_max = MinMax(df['target'])
    split_might = (target_max - target_min) / split
    split_point_values = pd.Series(np.linspace(target_min,target_max,split,endpoint=False))

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1
    
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, diabetes

#6 Cal Quant
#California housing Data Set with Classes for five quantiles of the target regression
def CaliforniaDataSetQuant():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    #print(california.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = california.data
    y_target = california.target
    attributes_names = california.feature_names
    class_names = ["very low", "low", "mid", "high", "very high"]
    split = len(class_names)
    
    #put x and y data together in a data frame to sort them and address a class (because regression problems have no class)
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)
    #sort by last column (target column)
    df.sort_values(by=[df.columns[-2]],inplace = True)
    #get the might of the split, at which rows the split happens, and which are the values of these split_points
    split_might = int(np.ceil(df.shape[0] / split))
    split_points = range(0  , df.shape[0] , split_might)
    split_point_values = df['target'][::split_might]

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1

    #resort data after the original index (so we just have the old data again with "new" classes)
    df.sort_index(inplace = True)
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, california

#7 Cal Abolut
#California housing Data Set with Classes for five subdivisions of the absolut values of the target regression
def CaliforniaDataSetAbsolut():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    #print(california.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = california.data
    y_target = california.target
    attributes_names = california.feature_names
    class_names = ["very low", "low", "mid", "high", "very high"]
    split = len(class_names)
    
    #put x and y data together in a data frame 
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)

    #get the might of the split and which are the values of these split_points
    target_min, target_max = MinMax(df['target'])
    split_might = (target_max - target_min) / split
    split_point_values = pd.Series(np.linspace(target_min,target_max,split,endpoint=False))

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1
    
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, california

#8
#Occupancy Detection Data Set
def OccupancyDataSet():
    from io import BytesIO
    from zipfile import ZipFile
    from urllib.request import urlopen
    #https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
    url = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip")

    #Download Zipfile and create pandas DataFrame
    zipfile = ZipFile(BytesIO(url.read()))
    #zipfile.printdir() 
    occupancy = pd.read_csv(zipfile.open('datatest2.txt'), )

    #Data without first column (date + time)
    x_data = occupancy[occupancy.columns[1:-1]].to_numpy()
    y_target = occupancy[occupancy.columns[-1]].to_numpy()
    attributes_names = occupancy.columns[1:-1].to_numpy()
    class_names = ["not occupied", "occupied"]
    
    return x_data, y_target, attributes_names, class_names, occupancy

#9 Cal Quant High Low
#California housing Data Set with Classes for five quantiles of the target regression
def CaliforniaHLDataSetQuant():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    #print(california.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = california.data
    y_target = california.target
    attributes_names = california.feature_names
    class_names = ["low", "high"]
    split = len(class_names)
    
    #put x and y data together in a data frame to sort them and address a class (because regression problems have no class)
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)
    #sort by last column (target column)
    df.sort_values(by=[df.columns[-2]],inplace = True)
    #get the might of the split, at which rows the split happens, and which are the values of these split_points
    split_might = int(np.ceil(df.shape[0] / split))
    split_points = range(0  , df.shape[0] , split_might)
    split_point_values = df['target'][::split_might]

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1

    #resort data after the original index (so we just have the old data again with "new" classes)
    df.sort_index(inplace = True)
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, california

#10 Cal Abolut High Low
#California housing Data Set with Classes for five subdivisions of the absolut values of the target regression
def CaliforniaHLDataSetAbsolut():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    #print(california.DESCR)
    
    #initialize some data from the DataSet and potential new classes
    x_data = california.data
    y_target = california.target
    attributes_names = california.feature_names
    class_names = ["low", "high"]
    split = len(class_names)
    
    #put x and y data together in a data frame 
    col = np.append(attributes_names,["target","target_new"])
    tar_old = np.atleast_2d(y_target).T
    tar_new = np.atleast_2d(np.zeros(x_data.shape[0])).T
    tar = np.append(tar_old, tar_new, axis = 1)
    df = pd.DataFrame(np.append(x_data,tar,axis = 1),columns=col)

    #get the might of the split and which are the values of these split_points
    target_min, target_max = MinMax(df['target'])
    split_might = (target_max - target_min) / split
    split_point_values = pd.Series(np.linspace(target_min,target_max,split,endpoint=False))

    #attribute the classes on the regression targets
    for i in range(split-1):
        df.loc[(df['target'] >= split_point_values.iloc[i]) & (df['target'] < split_point_values.iloc[i+1]) , 'target_new'] = i
    df.loc[df['target'] >= split_point_values.iloc[-1]  , 'target_new'] = split-1
    
    #x_data are all columns except the last which is the new y_target_class column
    x_data =df[df.columns[:-2]].to_numpy()
    y_target_class = df[df.columns[-1]].to_numpy()

    return  x_data, y_target_class, attributes_names, class_names, california


# In[2]:


#x, y, attributes, classes, full = IrisDataSet()
#x, y, attributes, classes, full = DiabetesDataSetQuant()
#x, y, attributes, classes, full = DiabetesDataSetAbsolut()
#x, y, attributes, classes, full = BostonHPDataSetQuant()
#x, y, attributes, classes, full = BostonHPDataSetAbsolut()
#x, y, attributes, classes, full = CaliforniaDataSetQuant()
#x, y, attributes, classes, full = CaliforniaDataSetAbsolut()
#x, y, attributes, classes, full = OccupancyDataSet()
#df = pd.DataFrame(np.append(x,np.atleast_2d(y).T,axis=1), columns=np.append(attributes,"target"))
#print(full.DESCR)
#print(df)


# In[3]:


#plt.scatter(df[df.columns[3]],df[df.columns[0]],c = df.target, s = 20 )
#plt.hist(full.target)
#print(full.target)
#plt.hist(y)


# In[4]:


#np.set_printoptions(edgeitems=10,infstr='inf',linewidth=150, nanstr='nan', precision=8,suppress=1, threshold=1000, formatter=None)


# In[ ]:




