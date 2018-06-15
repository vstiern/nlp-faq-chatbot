# faq_chatbot_4.0.py
# version 4 include selection of schools
# be able to ask continous questionss
### import libraries
import json
import math
import os
import pickle
import re
import time
import numpy as np
import pandas as pd
from random import sample
from fuzzywuzzy import process
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk libs
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


### define function to import correct FAQ (pre created faqs dictionary)
def faq_import(school_request):
    school_choices = {'A+ World Academy':'aplus_faq_dict',
                      'Hawaii Preparatory Academy (HPA)':'hpa_faq_dict',
                      'The American School in Switzerland (TASIS)':'tasis_faq_dict'}

    # execute fuzzy search to find top 1 similar choice
    school_match = process.extractOne(school_request, list(school_choices.keys()))

    # if math is below 50 ask user again
    while school_match[1] < 50:
        school_request_retry = input('Please respecify school. At this stage you can only query:\nA+ World Academy // Hawaii Preparatory Academy (HPA) // The American School in Switzerland (TASIS)\n')
        # execute fuzzy search to find top 1 similar choice
        school_match = process.extractOne(school_request_retry, list(school_choices.keys()))
        #print('school match: ' + str(school_match))
    # get school name
    school_select = school_match[0]
    #print('selected school: ' + str(school_select))
    # get corresponding faq dict name
    school_dict = school_choices[school_select]
    #print('selected dict: ' + str(school_dict))
    # define path to dict folder
    inpath = 'dict_texts/'
    # merge path and school dict
    filename = inpath + school_dict
    # open file
    infile = open(filename,'rb')
    # load dict
    faqs = pickle.load(infile)
    # close file
    infile.close()
    # return faq
    return faqs, school_select

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
    lemmas = [wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_result] # stem words using lemmatizer (verbs included)

    return lemmas

### define function for input question
def input_question(question, data, feats):
    # insert valid question into data (orginal faq questions)
    if question is not None:
        data.insert(0, question)

    # text process question and insert into feats (text processed faq questions)
    new_feats = text_process(question)
    print('### Question processed: ' + str(new_feats))
    feats.insert(0, new_feats)

    return data, feats

### define function to calculate cosine distances
def calculate_distances(feats):
    # vectorize org feats and input feat
    arr = np.array([])
    for i in range(len(feats)):
        arr = np.append(arr, ' '.join(feats[i]))

    # apply tf-idf to input and faq questions
    df2 = tfidf_vectorizer.fit_transform(arr)
    distances = cosine_similarity(df2[0:1], df2)
    #print(distances)
    return distances

### define fuzzy serach of orginal input and dict questions
def fuzzy_search(question):
    # execute fuzzy search to find top 1 similar choice
    fuzzy_faq_matches = process.extract(question, list(faqs.keys()))
    #print(fuzzy_faq_matches)

    fuzzy_faq_question = fuzzy_faq_matches[0][0]
    fuzzy_faq_score = fuzzy_faq_matches[0][1]

    #print('###' + str(fuzzy_faq_question1) + '###' + str(fuzzy_faq_score1))

    return fuzzy_faq_question, fuzzy_faq_score

### school contacts
school_contacts = {'A+ World Academy':'Email: aplus@piratelifetobe.rum\nPhone: 123-123-123',
                   'Hawaii Preparatory Academy (HPA)':'Email: hpa@gonesurfing.splash\nPhone: 213-213-213',
                   'The American School in Switzerland (TASIS)':'Email: tasis@tick.tock.käse\nPhone: 312-312-312'}

### define function to find most similar questions
def similarity_text(idx, distance_matrix, data, fuzzy_question, fuzzy_score):
    # find top5 similarity values, order them desc and return idx
    sorted_matrix = np.argsort(distance_matrix[0])[::-1][1:5]  #[:, ::-1]

    # subset get top 2 scores
    dist_score1 = distance_matrix[0][sorted_matrix[0]]
    dist_score2 = distance_matrix[0][sorted_matrix[1]]
    ## similarity thresholds
    # if first result is above 0.4 and not equal to second score print first result
    if ((dist_score1 > 0.4) & (dist_score1 != dist_score2)):
        print('\n' + faqs[data[sorted_matrix[0]]] + '\n')
    # else conduct fuzzy search of orginal input and dict questions
    elif fuzzy_score > 75:
        print('\n' + faqs[fuzzy_question] + '\n')
    # else request new questions
    else:
        print("I didn't find a suitable answer. Please try to respecify your question or contact the school:\n" + str(school_contacts[school_select]))

    print(distance_matrix[0][sorted_matrix[0]])

    # test mode
    #for idx in sorted_matrix:
        # get question from dict
        #org_quest = data[idx]
        #print(org_quest)
        #print(faqs[org_quest])
        # get score for org_quest
        #sim_score = distance_matrix[0][idx]
        #print(sim_score)

    #print(org_quest)
    #print('\n' + faqs[data[sorted_matrix[0]]] + '\n')

### define Q&A run function
def run(question):
    # import requested school faq
    data = list(faqs.keys())
    # process faq questions
    feats = [text_process(k) for k in data]
    # process input question
    input_results = input_question(question, data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]
    # calculate distance
    distance_matrix = calculate_distances(new_feats)
    # calculate fuzzy search matches
    fuzzy_result = fuzzy_search(question)
    fuzzy_question = fuzzy_result[0]
    fuzzy_score = fuzzy_result[1]

    # calculate similarity
    idx = 0
    similarity_text(idx, distance_matrix, new_data, fuzzy_question, fuzzy_score)

### define main frame for chatbot
if __name__ == "__main__":
    # query user for requested school
    school_request = input("What school would you like to know more about? For now you can chose from:\nA+ World Academy // Hawaii Preparatory Academy (HPA) // The American School in Switzerland (TASIS)\n")
    # import faq questions
    faq_results = faq_import(school_request)
    faqs = faq_results[0]
    school_select = faq_results[1]
    # keep question open until user quit application
    while True:
        question = input("What do you want to know about " + str(school_select) + "? [Note: Enter quit to exit app.] \n")
        # break if user inputs quit
        if(question.lower() == 'quit'):
            break
        run(question)
