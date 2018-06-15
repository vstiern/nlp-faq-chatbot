import math
import os
from random import sample
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import json
import numpy as np
from texttable import Texttable
import time
import indicoio
from functools import wraps
# Fuzzy Search Library
from fuzzywuzzy import fuzz
# py4j the bridge of Java and Python
from py4j.java_gateway import JavaGateway

# Your Indico API  : https://indico.io/
indicoio.config.api_key = "f60642b86fdd6872e7f5c47dfac0356a"

# Store Questions and Answers into Dictionary
faqs = {
    'Tell me a tongue twister' : 'Tell me a tongue twister',
    'What is your name': 'What is your name',
    'Can you show me the great barrier reef': 'Can you show me the great barrier reef',
    'How do I bring a school group to ACMI' : 'How do I bring a school group to ACMI',
    'How do I contact the Museum' : 'How do I contact the Museum',
    'What restaurants are at the Crown Entertainment Complex in Melbourne':'What restaurants are at the Crown Entertainment Complex in Melbourne',
    'Can you show me some of the offers you currently have at Crown':' Can you show me some of the offers you currently have at Crown',
    'How can I book a room at Crown Towers' : 'How can I book a room at Crown Towers',
    'What hotels can I stay at Crown Melbourne' : 'What hotels can I stay at Crown Melbourne',
    'Can you show me Crown Melbourne': 'Can you show me Crown Melbourne',
    'When are the fireballs on display':'When are the fireballs on display',
    'How do I play Baccarat' : 'How do I play Baccarat',
    'Can I gamble online with Crown Casino' : 'Can I gamble online with Crown Casino'
}

def make_feats(data):
    """
    Send our text data through the indico API and return each text example's text vector representation
    """
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # just a progress bar to show us how much we have left
    for chunk in tqdm(chunks):
        feats.extend(indicoio.text_features(chunk))

    return feats

def calculate_distances(feats):
    """
    cosine distance is the most reasonable metric for comparison of these 300d vectors
    """
    distances = cdist(feats, feats, 'cosine')
    return distances

def input_question(data, feats):
    """
    Receive the input questions by Users
    """
    # input a question
    question = input("What is your question? ")

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)
    return data, feats


def fuzzy_search(data,index):
    """
    When Confidence Threshold(Similarity) between 75% and 40%, we use fuzzy_search to check the how many word are similarity.
    """
    fuzzy_search_dict = {}
    for idx in index:
        value = fuzz.ratio(data[0],data[idx])
        fuzzy_search_dict[idx] = value
    return max(fuzzy_search_dict.items(), key=lambda x: x[1])[0]


def similarity_text(idx, distance_matrix, data, n_similar=5):
    """
    Use indici API and Fuzzy Search to select the answer for the Customer

    idx: the index of the text we're looking for similar questions to
         (data[idx] corresponds to the actual text we care about)
    distance_matrix: an m by n matrix that stores the distance between
                     document m and document n at distance_matrix[m][n]
    data: a flat list of text data
    """
    t = Texttable()
    t.set_cols_width([50, 20])

    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]


    #Fuzzy Search Dictionary Index
    fuzzy_ratio_array = sorted_distance_idxs.tolist()
    fuzzy_ratio_array.pop(0)
    print("fuzzy ratio array : %r" % fuzzy_ratio_array)

    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]
    print("sorted distance idx : %r" % most_sim_idx)
    # header for texttable
    t.add_rows([['Text', 'Similarity']])
    print(t.draw())

    # set the variable that will hold our matching FAQ
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        # actual text data for display
        datum = data[similar_idx]

        # distance in cosine space from our text example to the similar text example
        distance = distance_matrix[idx][similar_idx]

        # how similar that text data is to our input example
        similarity =  1 - distance

        # add the text + the floating point similarity value to our Texttable() object for display
        t.add_rows([[datum, str(round(similarity, 2))]])
        print(t.draw())

        # set a confidence threshold
        if similar_idx == most_sim_idx and similarity >= 0.75:
                    faq_match = data[most_sim_idx]
        elif similar_idx == most_sim_idx and similarity >= 0.4:
                    fuzzy_search_dict = {}
                    fuzzy_max_similarity_index = []
                    for idx in fuzzy_ratio_array:
                        value = fuzz.ratio(data[0],data[idx])
                        fuzzy_search_dict[idx] = value
                        fuzzy_max_similarity_index = max(fuzzy_search_dict.items(), key=lambda x: x[1])[0]

                    faq_match = data[fuzzy_max_similarity_index]
        else:
            sorry = "Sorry, I'm not sure how to respond. Let me find someone who can help you."

    # print the appropriate answer to the FAQ, or bring in a human to respond
    # TODO
    if faq_match is not None:
            print("A: %r" % faqs[faq_match])

    else:
            print(sorry)


def run():
    """
    Excute the Chatbot
    """
    data = list(faqs.keys())
    print("FAQ data received. Finding features.")
    feats = make_feats(data)
    with open('faq_feats.pkl', 'wb') as f:
        pickle.dump(feats, f, protocol = 2)
    print("FAQ features found!")

    with open('faq_feats.pkl', 'rb') as f:
            feats = pickle.load(f)
    print("Features found -- success! Calculating similarities...")

    input_results = input_question(data, feats)

    t0 = time.time()
    new_data = input_results[0]
    new_feats = input_results[1]

    distance_matrix = calculate_distances(new_feats)
    print("Similarities found. Generating table.")

    idx = 0

    similarity_text(idx, distance_matrix, new_data)
    t1 = time.time()
    print(t1-t0)

    print('\n' + '-' * 80)

run()
