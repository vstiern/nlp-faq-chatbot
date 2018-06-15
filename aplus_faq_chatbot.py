# Group Project - FAQ Chatbot
# 2018-06-12

# import libraries
import math
import os
import time
import json
import numpy as np
import indicoio
import pickle
from random import sample
from tqdm import tqdm
from scipy.spatial.distance import cdist
from texttable import Texttable # require cjkwrap: pip install cjkwrap

# insert API Key for indicoio
indicoio.config.api_key = "f60642b86fdd6872e7f5c47dfac0356a"

'''
Use indico's Text Features API to find text similarity and create a customer support bot that automatically responds to FAQs from users.
Tutorial: https://indico.io/blog/faqs-bot-text-features-api/
'''
# import faqs dictionary
inpath = '/Users/Mille/GDriveMBD/Term3/NLP/Group_Project/dict_texts/'
filename = inpath + 'aplus_faq_dict'
infile = open(filename,'rb')
faqs = pickle.load(infile)
infile.close()

def make_feats(data):
    """
    Send our text data throught the indico API and return each text example's text vector representation
    """
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # just a progress bar to show us how much we have left
    for chunk in tqdm(chunks):
        feats.extend(indicoio.text_features(chunk))

    return feats

def calculate_distances(feats):
    # cosine distance is the most reasonable metric for comparison of these 300d vectors
    distances = cdist(feats, feats, 'cosine')
    return distances

def similarity_text(idx, distance_matrix, data, n_similar=5):
    """
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
    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]

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
        else:
            sorry = "Sorry, I'm not sure how to respond. Let me find someone who can help you."

    # print the appropriate answer to the FAQ, or bring in a human to respond
    if faq_match is not None:
            print("A: %r" % faqs[faq_match])
    else:
            print(sorry)

def input_question(data, feats):
    # input a question
    question = input("What is your question? ")

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)

    return data, feats

def run():
    data = list(faqs.keys())
    print("FAQ data received. Finding features.")
    # open file - writing only in binary format
    feats = make_feats(data)
    print(feats)
    with open('faq_feats.pickle', 'wb') as f:
        pickle.dump(feats, f)
    print("FAQ features found!")
    # open file - reading only in binary format
    with open('faq_feats.pickle', 'rb') as f:
            feats = pickle.load(f)
    print("Features found -- success! Calculating similarities...")
    # get input results
    input_results = input_question(data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]
    # get distance
    distance_matrix = calculate_distances(new_feats)
    print(type(distance_matrix))
    print("Similarities found. Generating table.")
    # get similarity
    idx = 0
    similarity_text(idx, distance_matrix, new_data)
    print('\n' + '-' * 80)

if __name__ == "__main__":
    run()
