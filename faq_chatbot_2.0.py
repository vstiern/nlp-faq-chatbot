### import libraries
import re
import math
import os
import time
import json
import numpy as np
import pickle
import pandas as pd
from random import sample
from tqdm import tqdm
from scipy.spatial.distance import cdist
from texttable import Texttable # require cjkwrap: pip install cjkwrap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk libs
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

### import data - pre created faqs dictionary
inpath = '/Users/Mille/GDriveMBD/Term3/NLP/Group_Project/dict_texts/'
filename = inpath + 'aplus_faq_dict'
infile = open(filename,'rb')
faqs = pickle.load(infile)
infile.close()

### initialize text processing
# define CountVectorizer
#count_vectorizer = CountVectorizer(tokenizer=lambda doc:doc,
#                                   lowercase=False)
# define tf-idf vectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=False)

# define stop words
stop_words = set(stopwords.words('english'))
# define lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

### define text processing function for dict input
def text_process(data):
    '''
    function to transform raw dictionary into simplier form
    to calculate a more relvant vector similarity distance.
    '''
    only_letters = re.sub("[^a-zA-Z]", " ", data) # remove all but letters
    tokens = word_tokenize(only_letters) # tokenize words
    lower_case = [l.lower() for l in tokens] # convert to lowercase
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) # filter stopwords
    lemmas = [wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_result] # stem words using lemmatizer
    return lemmas

### define input question function
def input_question(data, feats):
    # input a question
    question = input("What is your question? ")
    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = text_process(question)
    print('-----question processed' + str(new_feats))
    feats.insert(0, new_feats)

    return data, feats

def calculate_distances(feats):
    # vectorize org feats and input feat
    #vect_feats = count_vectorizer.fit_transform(feats)
    arr = np.array([])
    for i in range(len(feats)):
        arr = np.append(arr, ' '.join(feats[i]))

        #print(' '.join(feats[i]))
    df2 = tfidf_vectorizer.fit_transform(arr)
    #print(df2.shape)
    distances = cosine_similarity(df2[0:1], df2)
    #print(distances)
    return distances

    #vect_feats = tfidf_vectorizer.fit_transform(feats)
    #print(vect_feats)
    # cosine distance is the most reasonable metric for comparison of these 300d vectors
    #istances = cdist(vect_feats, vect_feats, 'cosine')
    #return distances

def similarity_text(idx, distance_matrix, data, n_similar=5):
    """
    idx: the index of the text we're looking for similar questions to
         (data[idx] corresponds to the actual text we care about)
    distance_matrix: an m by n matrix that stores the distance between
                     document m and document n at distance_matrix[m][n]
    data: a flat list of text data
    """
    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    print(sorted_distance_idxs)

    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]
    # print answer
    print(faqs[data[most_sim_idx-1]])
    #print(distance_matrix[sorted_distance_idxs])

### define run function - combine all above
def run():
    data = list(faqs.keys())
    #print("FAQ data received. Finding features.")
    # open file - writing only in binary format
    feats = [text_process(k) for k in data]
    print(feats)

    #with open('faq_feats.pickle', 'wb') as f:
    #    pickle.dump(feats, f)
    #print("FAQ features found!")
    ## open file - reading only in binary format
    #with open('faq_feats.pickle', 'rb') as f:
    #        feats = pickle.load(f)
    #print("Features found -- success! Calculating similarities...")
    # get input results
    input_results = input_question(data, feats)
    # print
    new_data = input_results[0]
    new_feats = input_results[1]
    #print(new_data)
    #print(new_feats)

    # get distance
    distance_matrix = calculate_distances(new_feats)
    print(distance_matrix)
    # get similarity
    idx = 0
    similarity_text(idx, distance_matrix, new_data)
    print('\n' + '-' * 80)

if __name__ == "__main__":
    run()
