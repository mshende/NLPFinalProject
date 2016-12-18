#Author: Maya Shende
from __future__ import print_function

from enum import Enum
import pandas as pd
import numpy as np
from MayaUtils import Transform
from MayaUtils import process_file
from sklearn.feature_extraction.text import CountVectorizer
import math

class BoWType(Enum):
    #Option to select type of BoW

    binary = 0
    frequency = 1
    tfidf = 2

# word: the word for which we are filling in the term-doc-matrix
# frequency_dict: dictionary of dictionaries, key-value pairs are
#                 documents and dictionaries of word frequency for
#                 that doc
# doc_number: the document for which we are filling in the matrix
# num_docs: total number of documents
# inverse_doc_freq_dict: doctionary of number of docs containing each
#                        word
def compute_tfidf(word, frequency_dict, doc_number, num_docs, inverse_doc_freq_dict):
    doc_frequency = frequency_dict[doc_number]
    term_frequency = doc_frequency[word]
    inverse_doc_frequency = num_docs / inverse_doc_freq_dict[word]
    return(term_frequency * math.log(num_docs / inverse_doc_frequency))


def create_words_list(transform_option):
    output_tokens = process_file("train.tsv", transform_option)
    word_list = []
    for item in output_tokens:
        words = item[1]
        for word in words:
            if word not in word_list:
                word_list.append(word)
    return word_list


def create_bag_of_words(BoW_option, transform_option):
    word_list = create_words_list(transform_option)
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words=None,
                                 max_features=5000)
    train_data_features = vectorizer.fit_transform(word_list)
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()
    print("vocab: ", vocab)


create_bag_of_words(BoWType.frequency, Transform.tokens)
