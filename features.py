# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:30:24 2020

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
from fuzzywuzzy import fuzz

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

import os
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
import textacy.similarity

from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize
from gensim.similarities import MatrixSimilarity


stopwords = set(nltk_stopwords.words('english'))

# load different types of data
src = os.getcwd()+"\\"
df_train_orig= pd.read_csv(src+'quora_train.csv', delimiter = ',')
df_test_orig = pd.read_csv(src+'quora_test.csv', delimiter = ',')

df_train = pd.read_csv(src+'df_train_stem.csv', delimiter = ',')
df_test= pd.read_csv(src+'df_test_stem.csv', delimiter = ',')

df_train_le = pd.read_csv(src+'train_lemma.csv', delimiter = ',')
df_test_le= pd.read_csv(src+'test_lemma.csv', delimiter = ',')

df_train_clean = pd.read_csv(src+'train_fullclean.csv', delimiter = ',')
df_test_clean= pd.read_csv(src+'test_fullclean.csv', delimiter = ',')


# fill na values
df_train_orig = df_train_orig.fillna('')
df_test_orig = df_test_orig.fillna('')
df_train = df_train.fillna('')
df_test = df_test.fillna('')
df_train_le = df_train_le.fillna('')
df_test_le = df_test_le.fillna('')
df_train_clean = df_train_le.fillna('')
df_test_clean = df_test_le.fillna('')

# create features data frame
train_feat = pd.DataFrame()
test_feat = pd.DataFrame()

# feature: length of sentence, number of words, length of characters, different of these features

train_feat ['q1_len'] = df_train_orig.question1.map(lambda x: len(str(x)))
train_feat ['q2_len'] = df_train_orig.question2.map(lambda x: len(str(x)))
train_feat ['q1_wordnum'] = df_train_orig.question1.map(lambda x: len(str(x).split()))
train_feat ['q2_wordnum']  = df_train_orig.question2.map(lambda x: len(str(x).split()))
train_feat['q1_len_char'] = df_train_orig.question1.map(lambda x: len(''.join(str(x).split())))
train_feat['q2_len_char']  = df_train_orig.question2.map(lambda x: len(''.join(str(x).split())))
train_feat['char_diff'] = abs(train_feat['q1_len_char']-train_feat['q1_len_char'])
train_feat['len_diff'] = abs(train_feat ['q1_len']-train_feat ['q2_len'])
train_feat ['word_diff'] = abs(train_feat ['q1_wordnum'] -train_feat ['q2_wordnum'])

test_feat ['q1_len'] = df_test_orig.question1.map(lambda x: len(str(x)))
test_feat ['q2_len'] = df_test_orig.question2.map(lambda x: len(str(x)))
test_feat ['q1_wordnum'] = df_test_orig.question1.map(lambda x: len(str(x).split()))
test_feat ['q2_wordnum']  = df_test_orig.question2.map(lambda x: len(str(x).split()))
test_feat['q1_len_char'] = df_test_orig.question1.map(lambda x: len(''.join(str(x).split())))
test_feat['q2_len_char']  = df_test_orig.question2.map(lambda x: len(''.join(str(x).split())))
test_feat['char_diff'] = abs(test_feat['q1_len_char']-test_feat['q1_len_char'])
test_feat['len_diff'] = abs(test_feat ['q1_len']-test_feat ['q2_len'])
test_feat ['word_diff'] = abs(test_feat ['q1_wordnum'] -test_feat ['q2_wordnum'])


# split sentences into words
train_q1_words = df_train_orig.question1.map(lambda x: str(x).lower().split())
train_q2_words = df_train_orig.question2.map(lambda x: str(x).lower().split())
test_q1_words = df_test_orig.question1.map(lambda x: str(x).lower().split())
test_q2_words = df_test_orig.question2.map(lambda x: str(x).lower().split())

train_q1_words_s = df_train.question1.map(lambda x: str(x).lower().split())
train_q2_words_s = df_train.question2.map(lambda x: str(x).lower().split())
test_q1_words_s = df_test.question1.map(lambda x: str(x).lower().split())
test_q2_words_s = df_test.question2.map(lambda x: str(x).lower().split())


# the function to get the ratio of shared words
def shared_words(q1_words,q2_words):
    q1 = {}
    q2 = {}
    for word in q1_words:
        if word not in stopwords:
            q1[word] = q1.get(word, 0) + 1
    for word in q2_words:
        if word not in stopwords:
            q2[word] = q2.get(word, 0) + 1
    shared_words_q1 = sum([q1[word] for word in q1 if word in q2])
    shared_words_q2 = sum([q2[word] for word in q2 if word in q1])
    total = sum(q1.values()) + sum(q2.values())
    if total < 1e-6:
        return [0.]
    else:
        return [1.0 * (shared_words_q1 + shared_words_q2) / total]

# get number of shared words
tr_shared= list()
for i in range(len(train_q1_words_s)):
    shared = shared_words(train_q1_words[i],train_q2_words[i])
    tr_shared.append(shared)
    
te_shared= list()
for i in range(len(test_q1_words)):
    shared = shared_words(test_q1_words[i],test_q2_words[i])
    te_shared.append(shared)

train_feat['shared_words'] = tr_shared
test_feat['shared_words'] = te_shared

# number of shared nouns

train_feat_noun_q1 = df_train_le.question1.map(lambda x: [word for word, pos in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if pos[:1] in ['N']])
train_feat_noun_q2 = df_train_le.question2.map(lambda x: [word for word, pos in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if pos[:1] in ['N']])
train_feat['noun_num_q1'] =train_feat_noun_q1 .map(lambda x: len(x))
train_feat['noun_num_q2'] = train_feat_noun_q2.map(lambda x: len(x))


test_feat_noun_q1 = df_test_le.question1.map(lambda x: [word for word, pos in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if pos[:1] in ['N']])
test_feat_noun_q2 = df_test_le.question2.map(lambda x: [word for word, pos in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if pos[:1] in ['N']])
test_feat['noun_num_q1'] =test_feat_noun_q1.map(lambda x: len(x))
test_feat['noun_num_q2'] = test_feat_noun_q2.map(lambda x: len(x))


tr_noun_shared= list()
for i in range(len(train_feat_noun_q1)):
    shared = shared_words(train_feat_noun_q1[i],train_feat_noun_q2[i])
    tr_noun_shared.append(shared)
train_feat['noun_shared']=tr_noun_shared

te_noun_shared= list()
for i in range(len(test_feat_noun_q1)):
    shared = shared_words(test_feat_noun_q1[i],test_feat_noun_q2[i])
    te_noun_shared.append(shared)
test_feat['noun_shared']=te_noun_shared



#get TF-IDF scores
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf_txt = pd.Series(
    df_train_orig['question1'].tolist() + df_train_orig['question2'].tolist() + df_test_orig['question1'].tolist() +
    df_test_orig['question2'].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)
tr_tfidf_q1 =  df_train_orig.question1.map(lambda x: tfidf.transform([str(x)]).data)
tr_tfidf_q2 =  df_train_orig.question2.map(lambda x: tfidf.transform([str(x)]).data)
te_tfidf_q1 =  df_test_orig.question1.map(lambda x: tfidf.transform([str(x)]).data)
te_tfidf_q2 =  df_test_orig.question2.map(lambda x: tfidf.transform([str(x)]).data)

train_feat['q1_tfidf_sum'] = tr_tfidf_q1.map(lambda x: np.sum(x))
train_feat['q2_tfidf_sum'] = tr_tfidf_q2.map(lambda x: np.sum(x))
train_feat['q1_tfidf_mean'] = tr_tfidf_q1.map(lambda x: np.mean(x))
train_feat['q2_tfidf_mean'] = tr_tfidf_q2.map(lambda x: np.mean(x))
train_feat['q1_tfidf_len'] = tr_tfidf_q1.map(lambda x: len(x))
train_feat['q2_tfidf_len'] = tr_tfidf_q2.map(lambda x: len(x))

test_feat['q1_tfidf_sum'] = te_tfidf_q1.map(lambda x: np.sum(x))
test_feat['q2_tfidf_sum'] = te_tfidf_q2.map(lambda x: np.sum(x))
test_feat['q1_tfidf_mean'] = te_tfidf_q1.map(lambda x: np.mean(x))
test_feat['q2_tfidf_mean'] = te_tfidf_q2.map(lambda x: np.mean(x))
test_feat['q1_tfidf_len'] = te_tfidf_q1.map(lambda x: len(x))
test_feat['q2_tfidf_len'] = te_tfidf_q2.map(lambda x: len(x))


# dulplicate number of questions
dulp_num = {}

for index, row in df_train_orig.iterrows():
    q1 = str(row.question1).strip()
    q2 = str(row.question2).strip()
    dulp_num[q1] = dulp_num.get(q1, 0) + 1
    if q1 != q2:
        dulp_num[q2] = dulp_num.get(q2, 0) + 1
        
for index, row in df_test_orig.iterrows():
    q1 = str(row.question1).strip()
    q2 = str(row.question2).strip()
    dulp_num[q1] = dulp_num.get(q1, 0) + 1
    if q1 != q2:
        dulp_num[q2] = dulp_num.get(q2, 0) + 1

train_feat['q1_dulp_num'] = df_train_orig.question1.map(lambda x : dulp_num[str(x).strip()])
train_feat['q2_dulp_num'] = df_train_orig.question2.map(lambda x : dulp_num[str(x).strip()])
test_feat['q1_dulp_num'] = df_test_orig.question1.map(lambda x : dulp_num[str(x).strip()])
test_feat['q2_dulp_num'] = df_test_orig.question2.map(lambda x : dulp_num[str(x).strip()])
                                                np.nan_to_num(question2_vectors))]

    
# distance features    
# jaccard coefficient

def jaccard_similarity(s1, s2):
    # convert to vectors
    cv = CountVectorizer(tokenizer=lambda x: x.split())
    vectors = cv.fit_transform([s1,s2]).toarray()
    ret = cv.get_feature_names()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator

# jaccard coefficient of train set
tr_jacc_coef = list()
for index, row in df_train.iterrows():
    jacc = jaccard_similarity(row.question1,row.question2)
    tr_jacc_coef.append(jacc)
    
train_feat['jacc_coef']=tr_jacc_coef
# jaccard coefficient of test set
te_jacc_coef = list()
for index, row in df_test.iterrows():
    jacc = jaccard_similarity(row.question1,row.question2)
    te_jacc_coef.append(jacc)
test_feat['jacc_coef']=te_jacc_coef   
    
#jarowinkler
tr_jarowinkler = list()
for index, row in df_train.iterrows():
    jaro = Levenshtein.jaro_winkler(row.question1,row.question2)
    tr_jarowinkler.append(jaro)
    
train_feat['jarowinkler'] = tr_jarowinkler
te_jarowinkler = list()
for index, row in df_test.iterrows():
    jaro = Levenshtein.jaro_winkler(row.question1,row.question2)
    te_jarowinkler.append(jaro)
test_feat['jarowinkler'] = te_jarowinkler

#dice distance


tr_dice = list()
for i in range(len(train_q1_words_s)):
    total = len(train_q1_words_s[i])+ len(train_q2_words_s[i])
    same = len(set(train_q1_words_s[i])&set((train_q2_words_s[i])))
    if total <1e-6:
        tr_dice.append(0.0)
    else:
        tr_dice.append(2 * float(same) / (float(total)))

te_dice = list()
for i in range(len(test_q1_words_s)):
    total = len(test_q1_words_s[i])+ len(test_q2_words_s[i])
    same = len(set(test_q1_words_s[i])&set((test_q2_words_s[i])))
    if total <1e-6:
        te_dice.append(0.0)
    else:
        te_dice.append(2 * float(same) / (float(total)))




    
# token sort ratio
train_feat['token_sort_ratio'] = df_train.apply(lambda x: textacy.similarity.token_sort_ratio(x['question1'], x['question2']), axis = 1)
test_feat['token_sort_ratio'] = df_test.apply(lambda x: textacy.similarity.token_sort_ratio(x['question1'], x['question2']), axis = 1)
train_feat["fuzz_ratio"]            = df_train.apply(lambda x: fuzz.ratio(x["question1"], x["question2"]), axis=1)
train_feat["fuzz_partial_ratio"]    = df_train.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
test_feat["fuzz_ratio"]            = df_test.apply(lambda x: fuzz.ratio(x["question1"], x["question2"]), axis=1)
test_feat["fuzz_partial_ratio"]    = df_test.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    
 

# cosine similarity

df_train_clean = df_train_clean.fillna('NULL')
df_test_clean = df_test_clean.fillna('NULL') 

questions = df_train_clean.question1.tolist() + df_train_clean.question2.tolist() + df_test_clean.question1.tolist() + df_test_clean.question2.tolist()
tr_questions = df_train_clean.question1.tolist() + df_train_clean.question2.tolist()

dictionary = Dictionary(list(tokenize(question, errors='ignore')) for question in questions)
#
corpus = [dictionary.doc2bow(list(tokenize(question, errors = 'ignore'))) for question in questions]
tfidf = TfidfModel(corpus)


def cos_sim(question1, question2):
    tfidf1 = tfidf[dictionary.doc2bow(list(tokenize(question1, errors='ignore')))]
    tfidf2 = tfidf[dictionary.doc2bow(list(tokenize(question2, errors='ignore')))]
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])

tr_cossim = list()
for index, row in df_train_orig.iterrows():
    sim = cos_sim(row['question1'], row['question2'])
    tr_cossim.append(sim)
train_feat['cos_sim']=tr_cossim
te_cossim = list()
for index, row in df_test_orig.iterrows():
    sim = cos_sim(row['question1'], row['question2'])
    te_cossim.append(sim)
test_feat['cos_sim']=te_cossim
from scipy import spatial

# word to vectors
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('D:\\GoogleNews-vectors-negative300.bin', binary=True)
vocab = model.vocab

def get_vector(question):
    res =np.zeros([300])
    count = 0
    for word in word_tokenize(question):
        if word in vocab:
            res += model[word]
            count += 1
    return res/count 

def word2v(question1, question2):
    try:
        sim = 1 - spatial.distance.cosine(get_vector(question1),  get_vector(question2))
        return float(sim)
    except:
        return float(0)

tr_word2vec = list()
for index, row in df_train_orig.iterrows():
    sim = word2v(row['question1'], row['question2'])
    tr_word2vec.append(sim)
    
train_feat['word2vec']=tr_word2vec

te_word2vec  = list()
for index, row in df_test_orig.iterrows():
    sim = word2v(row['question1'], row['question2'])
    te_word2vec.append(sim)
test_feat['word2vec']=te_word2vec

#graph features


import networkx as nx

from itertools import combinations

# load data
src = os.getcwd()+"\\"

train_path = os.path.join(src+'quora_train.csv')
test_path = os.path.join(src+'quora_test.csv')
df_train = pd.read_csv(train_path, delimiter = ',').fillna('')
df_test = pd.read_csv(test_path, delimiter = ',').fillna('')

df_train = df_train[['id',"qid1","qid2","question1","question2","is_duplicate"]]
df_test = df_test[['id',"qid1","qid2","question1","question2","is_duplicate"]]

# acquire clique size 
len_train = df_train.shape[0]
# combine questions together
df = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)

# construct grapj with edges and nodes
G = nx.Graph()
edges = [tuple(x) for x in df[['question1', 'question2']].values]
G.add_edges_from(edges)

map_label = dict(((x[0],x[1]) for x in df[['question1', 'question2']].values))
map_clique_size = {}
# find cliques
cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))
for cli in cliques:
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_label.items():
            map_clique_size[q1, q2] = len(cli)
        elif (q2, q1) in map_label.items():
            map_clique_size[q2, q1] = len(cli)

df['clique_size'] = df.apply(lambda row: map_clique_size.get((row['question1'], row['question2']), -1), axis=1)


train_feat['clique_size'] = df[['clique_size']][:len_train]
test_feat['clique_size']= df[['clique_size']][len_train:]


# acquire pagerank for trainin data
qid_graph = {}
df_train.apply(lambda x: qid_graph.setdefault(x["question1"], []).append(x["question2"]), axis = 1)
df_train.apply(lambda x: qid_graph.setdefault(x["question2"], []).append(x["question1"]), axis = 1)

# select the parameters
MAX_ITER = 40
d = 0.85

pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
num_nodes = len(pagerank_dict)
iteration = {}

for iter in range(0, MAX_ITER):
    
    for node in qid_graph:    
        local_pr = 0
        
        for neighbor in qid_graph[node]:
            local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
        weight = (1-d)/num_nodes + d*local_pr
        pagerank_dict[node] = weight
        if weight in iteration:
            iteration[node].append(weight)
        else:
            iteration[node] = [weight]
    
tr_df = df_train.apply(lambda x: pd.Series({
        "pagerank_q1": pagerank_dict[x["question1"]],
        "pagerank_q2": pagerank_dict[x["question2"]]
    }), axis = 1)
train_feat['pagerank_1'], train_feat['pagerank_2']= tr_df.pagerank_q1*1e6, tr_df.pagerank_q2*1e6

#acquiring pagerank for test data
qid_graph = {}
df_test.apply(lambda x: qid_graph.setdefault(x["question1"], []).append(x["question2"]), axis = 1)
df_test.apply(lambda x: qid_graph.setdefault(x["question2"], []).append(x["question1"]), axis = 1)

MAX_ITER = 40
d = 0.85

pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
num_nodes = len(pagerank_dict)
iteration = {}

for iter in range(0, MAX_ITER):
    
    for node in qid_graph:    
        local_pr = 0
        
        for neighbor in qid_graph[node]:
            local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
        weight = (1-d)/num_nodes + d*local_pr
        pagerank_dict[node] = weight
        if weight in iteration:
            iteration[node].append(weight)
        else:
            iteration[node] = [weight]
    
df = df_test.apply(lambda x: pd.Series({
        "pagerank_q1": pagerank_dict[x["question1"]],
        "pagerank_q2": pagerank_dict[x["question2"]]
    }), axis = 1)

test_feat['pagerank_1'], test_feat['pagerank_2']= df.pagerank_q1*1e6, df.pagerank_q2*1e6


# unlist the data
train_feat.shared_words= train_feat.shared_words.map(lambda x: x.split('[')[1].split(']')[0])
test_feat.shared_words= test_feat.shared_words.map(lambda x: x.split('[')[1].split(']')[0])

train_feat.noun_shared= train_feat.noun_shared.map(lambda x: x.split('[')[1].split(']')[0])
test_feat.noun_shared= test_feat.noun_shared.map(lambda x: x.split('[')[1].split(']')[0])

# to csv files
train_feat.to_csv(src+'train_stat_feat.csv',index = False)
test_feat.to_csv(src+'test_stat_feat.csv', index = False)


