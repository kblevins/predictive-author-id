
# coding: utf-8

# In[1]:


import fasttext
import pandas as pd
import csv
model = fasttext.load_model('model.bin', label_prefix='__label__', encoding='utf-8')


# In[2]:


### Read our train data into a dataframe
train = pd.read_csv("train.csv")
copy = train.copy()
copy.head()
test = pd.read_csv("test.csv")
test_copy = test.copy()
test_copy.head()


# In[3]:


text = [text for text in test_copy['text']]
text[:3]


# In[26]:


with open('cleaned.txt', 'w+') as output:
    with open('train.csv', 'r+') as input:
        reader = csv.reader(input, delimiter=',', quotechar='"')
        next(reader, None)
        for row in reader:
            output.write(row[1] + ' __label__' + row[2] + '\n')


# In[27]:


classifier = fasttext.supervised('cleaned.txt', 'model')


# In[57]:


labels = classifier.predict_proba(text, k=1)
print (labels)


# In[78]:


final = pd.DataFrame(labels, columns = ['FastText Prediction'])
final.head()


# In[79]:


cols = ['FastText Prediction']
L = [pd.DataFrame(final[x].values.tolist(), columns=['Winning Author','Probability']) for x in cols]
final = pd.concat(L, axis=1, keys=cols)
final.head()


# In[77]:


labels_all = classifier.predict_proba(text, k=3)
print (labels_all)


# In[ ]:


# If we had a test file, we could measure the precision with: 

result = classifier.test('test.txt')
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)

