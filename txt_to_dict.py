import nltk
import os
import json
import pickle

# assign input file
inpath = '/Users/Mille/GDriveMBD/Term3/NLP/Group_Project/org_texts/'
infile = inpath + 'aplus_faq.txt'
# assign empty dict
d = {}
# open txt file
with open(infile) as f:
    # read file
    trial = f.read()
    # tokenize by line
    tokenize = nltk.line_tokenize(trial)
    # even numbers assign to key, odd to value in dict
    for i in range(0, len(tokenize), 2):
        d[tokenize[i]] = tokenize[i+1]

# assign output path & filename
outpath = '/Users/Mille/GDriveMBD/Term3/NLP/Group_Project/dict_texts/'
outname = outpath + 'aplus_faq_dict'
# write outfile
outfile = open(outname, 'wb')
pickle.dump(d,outfile)
outfile.close()



    # write dict to new file - json format
    #with open(output_name, 'wb') as file:
    #    file.write(json.dumps(d))
