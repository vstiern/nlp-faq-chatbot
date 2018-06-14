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

### define function for input question
def input_question(question, data, feats):
    # input a question
    #question = input("What is your question? ")

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = text_process(question)
    print('--question processed' + str(new_feats))
    feats.insert(0, new_feats)

    return data, feats

### define function to calculate cosine distances
def calculate_distances(feats):
    # vectorize org feats and input feat
    #vect_feats = count_vectorizer.fit_transform(feats)
    arr = np.array([])
    for i in range(len(feats)):
        arr = np.append(arr, ' '.join(feats[i]))

    df2 = tfidf_vectorizer.fit_transform(arr)
    distances = cosine_similarity(df2[0:1], df2)
    #print(distances)
    return distances

### define function to find most similar questions
def similarity_text(idx, distance_matrix, data, n_similar=5):
    """
    idx: the index of the text we're looking for similar questions to
         (data[idx] corresponds to the actual text we care about)
    distance_matrix: an m by n matrix that stores the distance between
                     document m and document n at distance_matrix[m][n]
    data: a flat list of text data
    """
    # find top5 similarity values
    # order them desc
    # return idx
    sorted_matrix = np.argsort(distance_matrix[0])[::-1][1:5]  #[:, ::-1]
    # modify idx -> remove input question
    #sorted_matrix_1 = sorted_matrix[:, 1::]

    # test mode
    for idx in sorted_matrix:
        # get question from dict
        org_quest = data[idx]
        #print(org_quest)
        #print(faqs[org_quest])
        # get score for org_quest
        sim_score = distance_matrix[0][idx]
        print(sim_score)

    #print(org_quest)
    print(faqs[data[sorted_matrix[0]]] + '\n')

    #print(type(sorted_matrix_1))
    # reshape numpyarr to workable format
    #good_shape = sorted_matrix_1.tolist()
    #print(good_shape[0:3])
    # return question behind index
    #question_of_index = data[sorted_matrix_1]
    #print(sorted_matrix_1[:,1])

    #print(faqs[data[sorted_matrix_1[:,1]]])
    # list them for comparision




    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    #sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    #print(sorted_distance_idxs)

    # this is the index of the text that is most similar to the query (index 0)
    #most_sim_idx = sorted_distance_idxs[1]
    # print answer
    #print(faqs[data[most_sim_idx-1]])
    #print(distance_matrix[sorted_distance_idxs])

### define run function - combine all above
def run(question):
    data = list(faqs.keys())
    feats = [text_process(k) for k in data]
    # get input results
    input_results = input_question(question, data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]
    # get distance
    distance_matrix = calculate_distances(new_feats)
    #print(distance_matrix)
    # get similarity
    idx = 0
    similarity_text(idx, distance_matrix, new_data)
    #print('\n' + '-' * 80)

if __name__ == "__main__":
    while True:
        question = input("What is your question? \nNote: Enter quit to exit \n")
        if(question.lower() == 'quit'):
            break
        run(question)
