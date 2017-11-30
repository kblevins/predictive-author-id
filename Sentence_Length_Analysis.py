
# coding: utf-8

# In[2]:


# read in some helpful libraries
import nltk # the natural langauage toolkit, open-source NLP
import pandas as pd # dataframes
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from collections import Counter

# initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[3]:


### Read our train data into a dataframe
texts = pd.read_csv("train.csv", encoding = 'latin-1')
texts.head()


# In[19]:


# make a copy of texts to work with
texts_len = texts

# add the sentence length for each row
texts_len['sentence_len'] = [len(str.split(s)) for s in texts['text']]
texts_len.head()


# In[46]:


# get summary data for each author on the sentence length
sentence_summary = pd.DataFrame(texts_len.groupby('author')['sentence_len'].describe())
sentence_summary


# In[62]:


# plot

plt.bar(np.arange(3), sentence_summary['mean'], yerr=sentence_summary['std'], color = ['red', 'blue', 'purple'], alpha = 0.6)
plt.xticks(np.arange(3), sentence_summary.index)
plt.title("Average Sentence Length by Author")
plt.xlabel("Author")
plt.ylabel("Sentence Length")
plt.show()

