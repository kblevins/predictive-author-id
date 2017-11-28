import nltk
import pandas as pd

texts = pd.read_csv("raw_data\\train.csv", encoding = 'latin-1')

eap = texts.loc[texts['author'] == 'EAP']
hpl = texts.loc[texts['author'] == 'HPL']
mws = texts.loc[texts['author'] == 'MWS']