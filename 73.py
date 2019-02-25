#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np   
import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re, string, timeit
from string import punctuation
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
import statsmodels.api as sm
from scipy.interpolate import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels as s
from numpy.polynomial import polynomial
import scipy
import math
from sys import stdout
from sklearn.preprocessing import PolynomialFeatures
import statsmodels as s
from numpy.polynomial import polynomial
import scipy
from sklearn import linear_model
import statsmodels.formula.api as smf
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


#set data
wiki= pd.read_csv('C:/Users/N_Solgi/Desktop/ArtificialIntelligence/ClusteringPractice/week02/people_wiki.csv')


# In[2]:


wiki.shape


# In[4]:


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    print (data[0],indices[0], indptr[0],shape[0])
    return csr_matrix( (data, indices, indptr), shape)


# In[5]:


word_count = load_sparse_csr('C:/Users/N_Solgi/Desktop/ArtificialIntelligence/ClusteringPractice/week02/people_wiki_word_count.npz')


# In[6]:


word_count


# In[8]:


print (word_count[0])


# In[10]:


import json
from scipy.sparse import csr_matrix 
with open('C:/Users/N_Solgi/Desktop/ArtificialIntelligence/ClusteringPractice/week02/people_wiki_map_index_to_word.json') as people_wiki_map_index_to_word:    
    map_index_to_word = json.load(people_wiki_map_index_to_word)


# In[11]:


len(map_index_to_word)


# In[12]:


wiki.head(2)


# In[13]:


from sklearn.neighbors import NearestNeighbors


# In[14]:


model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)


# In[15]:


wiki[wiki['name'] == 'Barack Obama']


# In[19]:


distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)
neighbors = pd.DataFrame(data={'distance':distances.flatten()},index=indices.flatten())
print (wiki.join(neighbors).sort_values('distance')[['name','distance']][0:10])


# In[20]:


neighbors.head(5)


# In[22]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


#10 nearest neighbors by performing the following query
#model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)


# In[29]:


def unpack_dict(matrix, map_index_to_word):
    #table = list(map_index_to_word.sort('index')['category'])
    # if you're not using SFrame, replace this line with
    table = sorted(map_index_to_word, key=map_index_to_word.get)
    
    
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in range(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)


# In[31]:


wiki['word_count'].head(5)


# In[32]:



"""
name = 'Barack Obama'
row = wiki[wiki['name'] == name]
dic = row['word_count'].iloc[0]
word_table = sorted(dic, key=dic.get, reverse=True)
print word_table
value_table = [dic[i] for i in word_table]
print value_table
print zip(word_table,value_table)
"""


# In[35]:


import ast


# In[ ]:




