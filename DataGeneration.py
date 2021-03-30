#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn 

from sklearn import tree


# In[9]:


def RandomRows(temp_array, temp_size, temp_seed):
    number_of_rows = temp_array.shape[0]
    np.random.seed(temp_seed + temp_size)
    if temp_size < number_of_rows:
        random_indices = np.random.choice(number_of_rows, size=temp_size, replace=False)
        random_rows = temp_array[random_indices, :]
    else:
        random_indices = np.random.choice(number_of_rows, size=temp_size, replace=True)
        random_rows = temp_array[random_indices, :]
    return random_rows
#x_RandomRows = RandomRows(x_train, 1000, seed)


# In[12]:


def MinMax_2d(temp_array):
    MinMax_temp_array = np.vstack((temp_array.min(0),temp_array.max(0)))
    return MinMax_temp_array

def RandomFromMinMaxUniform(MinMax_array, temp_size, temp_seed):
    x_uniform = np.zeros((temp_size, MinMax_array.shape[1]))
    np.random.seed(temp_seed + temp_size)
    for i in range(MinMax_array.shape[1]):
        iMin = MinMax_array[0,i]
        iMax = MinMax_array[1,i]

        x_uniform[:,i] = (iMax - iMin) * np.random.random_sample(temp_size) + iMin
    return x_uniform

#MinMax_x_train = MinMax_2d(x_train)
#x_RandomUniform = RandomFromMinMaxUniform(MinMax_x_train, 1000, seed)


# In[15]:


# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
#from sklearn.model_selection import GridSearchCV
def DensityApproximation(x_kernel_train, temp_size, temp_seed):
    np.random.seed(temp_seed + temp_size)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde.fit(x_kernel_train)
    x_SynthDist = kde.sample(temp_size, random_state=temp_seed)

    return x_SynthDist

#x_Density = DensityApproximation(x_train, 1000, seed)


# In[171]:


def MinMax_2d(temp_array):
    MinMax_temp_array = np.vstack((temp_array.min(0),temp_array.max(0)))
    return MinMax_temp_array

def RandomAroundSampleNormal(temp_sample, temp_MinMax_array, temp_size, temp_seed, temp_var):
    x_normal = np.zeros((temp_size, temp_MinMax_array.shape[1]))
    np.random.seed(temp_seed + temp_size)
    for i in range(temp_MinMax_array.shape[1]):
        iMin = temp_MinMax_array[0,i]
        iMax = temp_MinMax_array[1,i]
        iSigma = abs(temp_var * (iMax - iMin))
        x_normal[:,i] = np.random.normal(loc=temp_sample[i], scale=iSigma, size=temp_size)
        
    return x_normal
#x_RandomNormalSample = RandomAroundSampleNormal(x_train[0],MinMax_2d(x_train),500,seed, 0.15)


# In[24]:


#foil_classification is a function which asigns to the classes the corresponding fact and foil classes
def foil_classification_s(temp_y_array, temp_sample, blackbox, target, comment_bool):
    temp_sample_class = blackbox.predict([temp_sample])
    if temp_sample_class == target:
        print("Error! Target is same class as the sample, so target will be overwritten to -1")
        target = -1
        
    if target == -1:
        temp_foil_class = np.where(temp_y_array == temp_sample_class, 1, 0)
        if comment_bool == 1:
            print("Sample-Class is %s. Foil-Class is every other class" % (temp_sample_class))
    else:
        temp_foil_class = np.where(temp_y_array == target, 0, 1) 
        if comment_bool == 1:
            print("Sample-Class is %s. Foil-Class is  %s" % (temp_sample_class, target))
        
        
    temp_foil_class = np.atleast_2d(temp_foil_class).T
    return temp_foil_class

def foil_classification_c(temp_y_array, temp_sample_class, blackbox, target, comment_bool):
    if temp_sample_class == target:
        print("Error! Target is same class as the sample, so target will be overwritten to -1")
        target = -1
        
    if target == -1:
        temp_foil_class = np.where(temp_y_array == temp_sample_class, 1, 0)
        if comment_bool == True:
            print("Sample-Class is %s. Foil-Class is every other class" % (temp_sample_class))
    else:
        temp_foil_class = np.where(temp_y_array == target, 0, 1) 
        if comment_bool == True:
            print("Sample-Class is %s. Foil-Class is  %s" % (temp_sample_class, target))

    temp_foil_class = np.atleast_2d(temp_foil_class).T
    return temp_foil_class

#foil_classification_s(y_test_predict, temp_sample = x_test[0] , blackbox = NeuralTest, target = 0, comment_bool = True)


# In[25]:


from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import*

def normaleucdistancel1(temp_array, temp_sample):
    temp_complete_array = np.vstack([temp_array, temp_sample])
    temp_normal_array = normalize(temp_complete_array, norm='l1', axis = 0) 
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    
    return nedist

def normaleucdistancel2(temp_array, temp_sample):
    temp_complete_array = np.vstack([temp_array, temp_sample])
    temp_normal_array = normalize(temp_complete_array, norm='l2', axis = 0) 
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    
    return nedist

def normaleucdistancel1xi(temp_array, temp_sample, xi):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l1', axis = 0) * xi
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    
    return nedist

def normaleucdistancel2xi(temp_array, temp_sample, xi):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l2', axis = 0) * xi
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    
    return nedist

def localscoreweightnedistl1(temp_array, temp_sample, xi, yi):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l1', axis = 0)
    
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    #print(nedist)
    #print(pow(nedist, 2))
    local_nedist_score = 1/(1+pow(nedist , xi)  * pow(10, yi) )
    
    return local_nedist_score

def localscoreweightnedistl2(temp_array, temp_sample, xi, yi):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l2', axis = 0)
    
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    #print(nedist)
    #print(pow(nedist, 2))
    local_nedist_score = 1/(1+pow(nedist , xi)  * (pow(10, yi) * math.sqrt(temp_array.shape[0])) )
    
    return local_nedist_score

def localscoreweighteasynedistl1(temp_array, temp_sample):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l1', axis = 0)
    
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    #print(nedist)
    #print(pow(nedist, 2))
    local_nedist_score = 1/(1+nedist*100)
    
    return local_nedist_score
def localscoreweighteasynedistl2(temp_array, temp_sample):
    temp_complete_array = np.vstack([temp_array, temp_sample]) 
    temp_normal_array = normalize(temp_complete_array, norm='l2', axis = 0)
    
    nedist = euclidean_distances(temp_normal_array[0:-1], temp_normal_array[-1].reshape(1, -1))
    #print(nedist)
    #print(pow(nedist, 2))
    local_nedist_score = 1/(1+nedist*100)
    
    return local_nedist_score
# In[29]:


import math

def GenAlgo(temp_x_array, temp_x_sample, minmax_array, factfoil, blackbox, output_size, g, pc, pm, target):
    #initialize the part of generation 0 which has is fact or fail depending on what is the factfoil target
    x_gen_temp_factfoil = temp_x_array
    y_gen_temp_factfoil = blackbox.predict(x_gen_temp_factfoil)
    yff_gen_temp_factfoil = foil_classification_s(y_gen_temp_factfoil, temp_x_sample , blackbox = blackbox, target = target, comment_bool = False)
    x_gen0 = x_gen_temp_factfoil[np.squeeze(yff_gen_temp_factfoil) == factfoil,...]
    if x_gen0.shape[0] == 0:
        print("Error! No single example of the class %s is available in the input" % (factfoil))
        return False
    #initialize generation 0, repeat the input until more than output
    if x_gen0.shape[0] < output_size:
        i_repeat = math.ceil(1 / (x_gen0.shape[0] / output_size))
        x_gen0 = np.tile(x_gen0,(i_repeat,1))
    #declare temp generaton from gen 0 , temp fitness and probability from fitness 
    x_gen_temp = x_gen0
    x_gen_temp_fit = 1 - normaleucdistancel1(x_gen_temp, temp_x_sample)
    x_gen_temp_p = np.squeeze(x_gen_temp_fit) / np.sum(np.squeeze(x_gen_temp_fit))
    #go through each generation
    for i_Gen in range(g):
    #for i_Gen in range(1):
        #Crossover
        #get size of parents for crossover from pc
        cross_size = math.ceil(x_gen_temp.shape[0] * pc *0.5) * 2
        #get parents with roulette method
        x_potpar_indices = np.random.choice(x_gen_temp.shape[0], size=cross_size, replace=True, p=x_gen_temp_p)
        x_potpar = x_gen_temp[x_potpar_indices, :]
        #get which two features to cross through random permutation of the features and takeing the first two
        len_x_potpar = x_potpar.shape[0]
        perm_mat = np.zeros((len_x_potpar, x_potpar.shape[1]))
        for i in range(len_x_potpar):
            perm_mat[i,:] = np.random.permutation(x_potpar.shape[1])
        perm_mat = perm_mat.astype(int)
        #the random drawn potential parents cross theire respective features in pairs in order in wich they were drawn
        for i in range(0,x_potpar.shape[0],2):
            A_Oben = x_potpar[i,perm_mat[i,0]]
            B_Oben = x_potpar[i,perm_mat[i,1]]
            x_potpar[i,perm_mat[i,0]] = x_potpar[i+1,perm_mat[i,0]]
            x_potpar[i,perm_mat[i,1]] = x_potpar[i+1,perm_mat[i,1]]
            x_potpar[i+1,perm_mat[i,0]] = A_Oben
            x_potpar[i+1,perm_mat[i,1]] = B_Oben
        
        #Mutation
        #get size of clones for mutation from pm
        muta_size = math.ceil(x_gen_temp.shape[0] * pm )
        #get clones with roulette method
        x_potclo_indices = np.random.choice(x_gen_temp.shape[0], size=muta_size, replace=True, p=x_gen_temp_p)
        x_potclo = x_gen_temp[x_potclo_indices, :]
        #get random mutation for each feature for each potential clone. Mutation is between -1 and 1 times 0.2 of the absolute feature range
        minmax_diag = np.diag(minmax_array[1]-minmax_array[0])
        x_potclo_random_mut = (np.random.random_sample([muta_size,x_gen_temp.shape[1]]) * 2 ) - 1
        x_potclo_random_mut = np.matmul(x_potclo_random_mut , minmax_diag) * 0.1
        
        #get which two features to mutate through random permutation of the features and takeing the first two
        len_x_potclo = x_potclo.shape[0]
        perm_mat = np.zeros((len_x_potclo, x_potclo.shape[1]))
        for i in range(len_x_potclo):
            perm_mat[i,:] = np.random.permutation(x_potclo.shape[1])
        perm_mat = perm_mat.astype(int)
        #the random drawn potential parents cross theire respective features in pairs in order in wich they were drawn
        for i in range(0,x_potclo.shape[0]):
            x_potclo[i,perm_mat[i,0]] = x_potclo[i,perm_mat[i,0]] + x_potclo_random_mut[i,perm_mat[i,0]]
            x_potclo[i,perm_mat[i,1]] = x_potclo[i,perm_mat[i,1]] + x_potclo_random_mut[i,perm_mat[i,1]]
        
        #Old candidate pool + Crossover + Mutation - New points which have now false class
        x_gen_temp_factfoil = np.vstack([x_gen_temp, x_potpar, x_potclo])
        y_gen_temp_factfoil = blackbox.predict(x_gen_temp_factfoil)
        yff_gen_temp_factfoil = foil_classification_s(y_gen_temp_factfoil, temp_x_sample , blackbox = blackbox, target = target, comment_bool = False)
        x_gen_temp_toobig = x_gen_temp_factfoil[np.squeeze(yff_gen_temp_factfoil) == factfoil,...]
        
        #If new pool is to big (more than 5 times the output size, delete random which are to much with probability from the fitness)
        len_x_gen_temp_toobig = x_gen_temp_toobig.shape[0]
        if len_x_gen_temp_toobig > 5 * output_size:
            toomuch = len_x_gen_temp_toobig - 5 * output_size 
            x_gen_temp_toobig_fit = normaleucdistancel1(x_gen_temp_toobig, temp_x_sample)
            x_gen_temp_toobig_p = np.squeeze(x_gen_temp_toobig_fit) / np.sum(np.squeeze(x_gen_temp_toobig_fit))
            
            x_gen_bin_indices = np.random.choice(x_gen_temp_toobig.shape[0], size=toomuch, replace=False, p=x_gen_temp_toobig_p)
            x_gen_temp = np.delete(x_gen_temp_toobig, x_gen_bin_indices, 0)
        else:
            x_gen_temp = x_gen_temp_toobig
            
        x_gen_temp_fit = 1 - normaleucdistancel1(x_gen_temp, temp_x_sample)
        x_gen_temp_p = np.squeeze(x_gen_temp_fit) / np.sum(np.squeeze(x_gen_temp_fit))
    
    #the final output draws again random from the last gen with probability from the fitness of the samples
    x_gen_final_indices = np.random.choice(x_gen_temp.shape[0], size=output_size, replace=False, p=x_gen_temp_p)
    x_gen = x_gen_temp[x_gen_final_indices, :]
    
    
    return x_gen

#x_GenAlgo_foil = GenAlgo(x_RandomNormalSample, x_sample, MinMax_2d(x_train), 0, NeuralTest, 10000, 20, 0.3 ,0.5, target = tar)
#x_GenAlgo_fact = GenAlgo(x_RandomNormalSample, x_sample, MinMax_2d(x_train), 1, NeuralTest, 10000, 7, 0.3 ,0.5, target = tar)

