
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


line = "(SBARQ (WHNP (WP Best))(SQ (AUX sad)(NP (NNP happy))(VP (VB bad)(PP (IN superb))))(. ?))"

words = re.sub(r'\(\S*', '', line)
words = re.sub(r'\)', '', words)
print(line)

# Split on whitespace
wordList = words.split()

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


print(line)
