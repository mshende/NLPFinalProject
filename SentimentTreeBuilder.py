
#
# This file will build a sentiment tree based on the sentiment score from senti-wordnet
#

#
# breakdown = swn.senti_synset('breakdown.n.03')
#
# 'breakdown' = word you need scores for.
# 'n' = part of speech
# '03' = Usage (01 for most common usage and a higher number would indicate lesser common usages)
#

#
# n - NOUN
# v - VERB
# a - ADJECTIVE
# s - ADJECTIVE SATELLITE
# r - ADVERB
#

import re

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from string import punctuation

punctuation_to_remove = punctuation.replace("-", "")
punctuation_to_remove = punctuation_to_remove.replace("'", "")
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation_to_remove)

#line = "1 |BT| (ROOT (S (NP (DT The) (NNS performances)) (VP (VBP are) (NP (DT an) (JJ absolute) (NN joy))) (. .))) |ET|"

# Open the sentitree.txt file for writing
sentitree = open('sentitree.txt', 'w')

# Open the data.txt file for reading
data_text_file = open('data.txt', 'r')

# Load the data.txt lines into a list
data_text = data_text_file.readlines()

for line in data_text:
    words = re.sub(r'\S*\s*\|BT\|', '', line)
    words = re.sub(r'\|ET\|', '', words)
    words = re.sub(r'\(\S*', '', words)
    words = re.sub(r'\)', '', words)
    words = strip_punctuation(words)

    # Split on whitespace
    wordList = words.split()

    # sort the word list by length descending, so that the longer words get replaced first
    # this keeps smaller words contained within larger words from being replaced
    wordList.sort(key=len, reverse=True)

    for word in wordList:
        positive = 0
        negative = 0
        neutral = 0
        count = 0

        # For each sentiment in the synset add to the positive, negative, or neutral counts
        for syn in wn.synsets(word):
            count = count + 1
            senti_synset = swn.senti_synset(syn.name())
            positive = positive + senti_synset.pos_score()
            negative = negative + senti_synset.neg_score()
            neutral = neutral + senti_synset.obj_score()

        # if the count is greater than zero then normalize the values
        if count > 0:
            positive = positive / count
            negative = negative / count
            neutral = neutral / count

        # Replace the word with the positive or negative value
        regex = r'' + re.escape(word)
        if ((count > 0) & (positive > negative) & (positive > neutral)):
            line = re.sub(regex, 'P', line)
        elif ((count > 0) & (negative > positive) & (negative > neutral)):
            line = re.sub(regex, 'N', line)
        else:
            line = re.sub(regex, 'U', line)

    sentitree.write(line)
