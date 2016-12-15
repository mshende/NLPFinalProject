
#
# This file will build a sentiment tree based on the sentiment score from senti-wordnet
#
#
# The output will be used for input into a tree kernel.
#

# Method for creating a sentiment tree (BOS tree), used to simulate the bag-of-sentiments.
# We first read in the parse tree representations of sentences from a data file.
# For each parse tree we remove the words from the parse tree and add them to a list.
# For each word in the word list we use WordNet to obtain all of the senses for the word.
# This is done by getting the sets of synonyms called synsets for the word. Each sense of the word in the synset
# is then entered into the sentiwordnet senti_synset method to obtain the pos_score, neg_score, and obj_score of the
# word sense. Two different methods of calculating the resulting positivity, negativity, or neutrality of the word were tested.
# The first method was to sum all of the positive scores, negative scores, and neutral scores and divide them by the
# number of word senses. The second method was to use the first, and most common, word sense and use that score
# for the resulting positive, negative, or neutral word score.
# The positive words are replaced in the parse tree representation of the sentence with a "P".
# The negative words are replaced in the parse tree representation of the sentence with a "N".
# The neutral words are replaced in the parse tree representation of the sentence with a "U".
# The resulting new sentiment tree is then output to a file.
# The sentiment trees are then added to the existing feature vector, or used as an independent feature vector in the
# operation of the Tree Kernel.
#
#FORM:
# -1 |BT| (ROOT (S (S (NP (DT This)) (VP (VBZ is) (NP (DT a) (JJ heartfelt) (NN story)))) (: ...) (S (NP (PRP it))
# (ADVP (RB just)) (VP (VBZ is) (RB n't) (ADVP (DT a) (RB very)) (VP (VBG involving) (NP (CD one))))) (. .)))
# |BT| (BOP (U *) (U *) (U *) (U *) (P *) (P *) (U *) (U *) (U *) (P *) (U *) (P *) (U *) (U *) (U *)) |ET|
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
#print(line)

# Open the sentitree.txt file for writing
sentitree = open('sentitreedata.txt', 'w')

# Open the data.txt file for reading
data_text_file = open('data.txt', 'r')

# Load the data.txt lines into a list
data_text = data_text_file.readlines()

# For each line in the data file
for line in data_text:
    line = line.rstrip()                        # Remove the new line character
    line = words = re.sub(r'\|ET\|', '', line)  # Remove the end tree tag from the line

    words = re.sub(r'\S*\s*\|BT\|', '', line)   # Remove the begin tree tag
    words = re.sub(r'\(\S*', '', words)         # Remove the ( characters
    words = re.sub(r'\)', '', words)            # Remove the ) characters
    #words = strip_punctuation(words)           # Remove punctuation

    line += ' |BT| (BOP'  # append a begin tree tag and BOP on the line

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
        if ((count > 0) & (positive > negative) & (positive > neutral)):
            line += ' (P *)'
        elif ((count > 0) & (negative > positive) & (negative > neutral)):
            line += ' (N *)'
        else:
            line += ' (U *)'


    line += ') |ET|\n'
    #print(line)
    sentitree.write(line)
