# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:46:01 2020

@author: DELL
"""
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
import en_core_web_md
import sematch
import gensim
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import pylab as pl
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from tqdm import tqdm, tqdm_notebook
from nltk.corpus import stopwords as nltk_stopwords
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
#from cleaning_utils import *
from keras.preprocessing.text import Tokenizer
import warnings
warnings.filterwarnings('ignore')

from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec

from gensim.models import KeyedVectors
from sklearn.metrics import classification_report 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import itertools
import keras.backend 
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.optimizers import Adadelta

from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score 
src = os.getcwd()+"\\"


features_train = pd.read_csv(src+'train_stat_feat.csv')
features_test = pd.read_csv(src+'test_stat_feat.csv')
features_train = features_train.astype(np.float).fillna(0.0)
features_test = features_test.astype(np.float).fillna(0.0)

# logistic regression
from sklearn.linear_model import LogisticRegression 
import datetime
x_train = features_train
y_train = df_train.is_duplicate.fillna('')
x_test = features_test.fillna('')
y_test = df_test.is_duplicate.fillna('')

def pyplot_performance(y,name):
    x = [int(i) for i in range(1,6)]
    pl.figure(1)
    pl.ylabel(u'Accuracy')
    pl.xlabel(u'times')
    pl.plot(x,y,label=name)
    pl.legend()
    
def log_reg(x_train,y_train,x_test,y_test):
    starttime = datetime.datetime.now()
    clf1 = LogisticRegression()
    score1 = model_selection.cross_val_score(clf1,x_train,y_train,cv=5,scoring="accuracy")
    pyplot_performance(score1,"LogisticRegression")
    print (np.mean(score1))
    clf1.fit(x_train, y_train)
    y_true = y_test  
    y_pred = clf1.predict(x_test)  
    y_pred_pro = clf1.predict_proba(x_test)  
    y_scores = pd.DataFrame(y_pred_pro, columns=clf1.classes_.tolist())[1].values  
    print(classification_report(y_true, y_pred)) 
    print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
    pyplot_roc(y_true, y_scores,"LogisticRegression")
    endtime = datetime.datetime.now()
    print ("runtime:"+str((endtime - starttime).seconds / 60))

def pyplot_roc(y_true, y_scores,title):
    auc_value = roc_auc_score(y_true, y_scores) 
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)  
    pl.figure(2) 
    pl.plot(fpr, tpr, label=title+' (area = %0.4f)' % auc_value)  
    pl.plot([0, 1], [0, 1])  
    pl.xlim([0.0, 1.0])  
    pl.ylim([0.0, 1.05])  
    pl.xlabel('False Positive Rate')  
    pl.ylabel('True Positive Rate')  
    pl.title('ROC '+title)  
    pl.legend(loc="lower right")  
    
log_reg(x_train,y_train,x_test,y_test)



# gradient boosting



def gradient_boosting(x_train,y_train,x_test,y_test):
    print("gradient_boosting...")     
    starttime = datetime.datetime.now()
    clf = GradientBoostingClassifier()
    score = model_selection.cross_val_score(clf,x_train,y_train,cv=5,scoring="accuracy")
    print ('The accuracy of GradientBoosting:')
    print (np.mean(score))
    pyplot_performance(score,"GradientBoosting")
    
    clf.fit(x_train, y_train)
    y_true = y_test
    y_pred = clf.predict(x_test)  
    y_pred_pro = clf.predict_proba(x_test)  
    y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  
    print(classification_report(y_true, y_pred)) 
    print ("accuracy_score:",metrics.accuracy_score(y_true, y_pred))
    pyplot_roc(y_true, y_scores,"GradientBoosting")
    endtime = datetime.datetime.now()
    print ("runtime:"+str((endtime - starttime).seconds / 60))
    print("Begining prediction...")

gradient_boosting(x_train,y_train,x_test,y_test)


# lstm


src = os.getcwd()+"\\"

train_path = os.path.join(src+'df_train_stem.csv')
test_path = os.path.join(src+'df_test_stem.csv')
df_train = pd.read_csv(train_path, delimiter = ',')
df_test = pd.read_csv(test_path, delimiter = ',')

df_train = df_train.fillna('')
df_test = df_test.fillna('')

stops = set(nltk_stopwords.words('english'))


# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  
# load GoogleNews-vectors-negative300.bin.gz
word2vec = KeyedVectors.load_word2vec_format(src+'GoogleNews-vectors-negative300.bin.gz', binary=True)

questions_cols = ['question1', 'question2']

# Iterate over the questions only of both training and test datasets
for dataset in [df_train, df_test]:
    for index, row in dataset.iterrows():

        for question in questions_cols:

            q2n = [] 
            for word in str(row[question]).split():

                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            dataset.set_value(index, question, q2n)
            
# select parameters
# get the embedding matrix
embedding_dim = 300 
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  
embeddings[0] = 0  

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

max_seq_length = max(df_train.question1.map(lambda x: len(x)).max(),
                     df_train.question2.map(lambda x: len(x)).max(),
                     df_test.question1.map(lambda x: len(x)).max(),
                     df_test.question2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = int(0.25*len(df_train))
training_size = len(df_train) - validation_size

X = df_train[questions_cols]
Y = df_train['is_duplicate']


# get the training, validation set 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
feat_train = features_train.iloc[:len(X_train.question1)]
feat_val = features_train.iloc[len(X_train.question2):]
# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': df_test.question1, 'right': df_test.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

for dataset, side in itertools.product([X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


# Model parameters
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 15

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return keras.backend.exp(-keras.backend.sum(keras.backend.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')
feature_input = Input(shape = (features_train.shape[1],), dtype="float32")


embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

shared_lstm = LSTM(n_hidden)


left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Packraw data and features up for training
malstm = Model([left_input, right_input,feature_input], [malstm_distance])

optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# start training

malstm_trained = malstm.fit([X_train['left'], X_train['right'],feat_train], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right'], feat_val], Y_validation))

# plot the accuracy rate with epoch number
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# plot the model loss with epoch number
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# predict the test file
preds = malstm.predict([X_test['left'],X_test['right'],features_test])
preds = np.argmax(preds,axis=1)

preds = pd.DataFrame({"test_id": df_test["id"], "is_duplicate": df_test['is_duplicate'],"pred": preds})
# plot the ROC curve
def pyplot_roc(y_true, y_scores,title):
    auc_value = roc_auc_score(y_true, y_scores) 
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)  
    pl.figure(2) 
    pl.plot(fpr, tpr, label=title+' (area = %0.4f)' % auc_value)  
    pl.plot([0, 1], [0, 1])  
    pl.xlim([0.0, 1.0])  
    pl.ylim([0.0, 1.05])  
    pl.xlabel('False Positive Rate')  
    pl.ylabel('True Positive Rate')  
    pl.title('ROC '+title)  
    pl.legend(loc="lower right")  
    
pyplot_roc(df_test.is_duplicate, preds, "LSTM")


