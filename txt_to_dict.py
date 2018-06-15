import nltk
import os
import json
import pickle

# assign input path and file name
inpath = 'org_texts/'
infile = inpath + 'tasis_faq.txt'
# assign empty dict
d = {}
# open txt file
with open(infile) as f:
    # read file
    trial = f.read()
    # tokenize by line
    tokenize = nltk.line_tokenize(trial)
    # even numbers(questions) assign to key, odd (answers) to value in dict
    for i in range(0, len(tokenize), 2):
        d[tokenize[i]] = tokenize[i+1]

# assign output path and filename
outpath = 'dict_texts/'
outname = outpath + 'tasis_faq_dict'
# write outfile
outfile = open(outname, 'wb')
pickle.dump(d,outfile)
outfile.close()
