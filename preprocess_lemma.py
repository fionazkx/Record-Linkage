# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:05:03 2020

@author: DELL
"""


import sematch

import numpy as np
import pandas as pd
import itertools as it
import pickle
import glob
import os
import string
import gc
import re
import time
import nltk
import spacy
import textacy
import gensim
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from tqdm import tqdm, tqdm_notebook
from scipy import sparse
from scipy.optimize import minimize
from nltk.corpus import stopwords as nltk_stopwords
import en_core_web_md

from tqdm import tqdm

from nltk.stem import SnowballStemmer

import os
import random
import spacy

stemmer = SnowballStemmer('english')

# load train and test data tables

src = os.getcwd()
train_path = os.path.join(src+'\\quora_train.csv')
test_path = os.path.join(src+'\\quora_test.csv')
df_train = pd.read_csv(train_path, delimiter = ',')
df_test = pd.read_csv(test_path, delimiter = ',')

# extract useful columns
df_train = df_train[['id',"qid1","qid2","question1","question2","is_duplicate"]]
df_test = df_test[['id',"qid1","qid2","question1","question2","is_duplicate"]]


# fill the empty value

df_train = df_train.fillna('')
df_test = df_test.fillna('')


# load the corups 
nlp = spacy.load("en_core_web_md")



# clean the text

def clean_text(text):

    # unit
    text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)        
    text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)        
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"dollars", " dollar ", text)
    test = re.sub(r"(\d+)%", lambda m: m.group(1) + ' percent ', text )
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    
    # mathematical
    text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)          
    text = re.sub(r",000", '000', text)# 
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    
    # proper noun 
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r"pokemons", "pokemon", text)
    text = re.sub(r"pokémon", "pokemon", text)
    text = re.sub(r"pokemon go ", "pokemon-go ", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c \+\+", "cplusplus", text)
    text = re.sub(r"c \+ \+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    text = re.sub(r"f#", "fsharp", text)
    text = re.sub(r"g#", "gsharp", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"the european union", "eu", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" quaro ", " quora ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r"insidefacebook", "inside facebook", text)
    text = re.sub(r"donald trump", "trump", text)
    text = re.sub(r"the big bang", "big-bang", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"the european union", " eu ", text)


    # punctuation
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\|", " | ", text)


    # symbol replacement
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"₹", " rs ", text)     

    # symbol deletion
    test = re.sub('[\(\)\!\^\'\"\.;,+-\?\{\}\[\]\\/]', ' ', string)    
    # remove extra space
    text = ' '.join(text.split())

    return text


df_train['question1'] = df_train.question1.map(lambda x: clean_text(str(x).lower()))
df_train['question2'] = df_train.question2.map(lambda x: clean_text(str(x).lower()))


df_test['question1'] = df_test.question1.map(lambda x: clean_text(str(x).lower()))
df_test['question2'] = df_test.question2.map(lambda x: clean_text(str(x).lower()))


#remove stop words

stopwords = set(nltk_stopwords.words("english"))

df_train['question1'] = df_train.question1.map(lambda x: ' '.join([w for w in x.split(' ') if w not in stopwords]))
df_train['question2'] = df_train.question2.map(lambda x: ' '.join([w for w in x.split(' ') if w not in stopwords]))

df_test['question1'] = df_test.question1.map(lambda x: ' '.join([w for w in x.split(' ') if w not in stopwords]))
df_test['question2'] = df_test.question2.map(lambda x: ' '.join([w for w in x.split(' ') if w not in stopwords]))


#'Lemmatizing text.')

SYMBOLS = set(' '.join(string.punctuation).split(' ') + ['...', '“', '”', '\'ve'])
q1 = []
for doc in tqdm(nlp.pipe(df_train['question1'], n_threads=8, batch_size=10000)):
    word_list = ([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])
    q1.append(' '.join(i for i in word_list))
q2 = []
for doc in tqdm(nlp.pipe(df_train['question2'], n_threads=8, batch_size=10000)):
    word_list = ([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])
    q2.append(' '.join(i for i in word_list))
    
df_train.question1 = pd.DataFrame(q1)
df_train.question2 = pd.DataFrame(q2)

q1 = []
for doc in tqdm(nlp.pipe(df_test['question1'], n_threads=8, batch_size=10000)):
    word_list = ([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])
    q1.append(' '.join(i for i in word_list))
q2 = []
for doc in tqdm(nlp.pipe(df_test['question2'], n_threads=8, batch_size=10000)):
    word_list = ([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])
    q2.append(' '.join(i for i in word_list))
    
df_test.question1 = pd.DataFrame(q1)
df_test.question2 = pd.DataFrame(q2)




# output

df_train.to_csv(src+'\\train_lemma.csv', index = False)
df_test.to_csv(src+'test_lemma.csv', index = False)
