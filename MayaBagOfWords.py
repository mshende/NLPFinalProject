#Author: Maya Shende
from __future__ import print_function

from enum import Enum
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
from MayaUtils import Transform
from MayaUtils import process_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math

class BoWType(Enum):
    #Option to select type of BoW

    binary = 0
    frequency = 1
    tfidf = 2

def create_words_list(transform_option):
    print('creating output tokens list')
    output_tokens = process_file("train.tsv", transform_option)
    word_list = []
    sentiment_list = []
    for item in output_tokens:
    #     words = item[1]
        # for word in words:
        #     if word not in word_list:
        #         word_list.append(word)
        word_list.append(item[1])
        sentiment_list.append(item[0])
    return word_list, sentiment_list

def create_test_list():
    sentence1 = 'This is a review sentence'
    sentence2 = 'This is another review sentence'
    sentence3 = 'Test of the bag of words'

    regular_expression_tokenizer = RegexpTokenizer(r'\w+')

    clean_train_reviews = [sentence1, sentence2, sentence3]

    # for word in regular_expression_tokenizer.tokenize(sentence1):
    #     clean_train_reviews.append(word)

    # for word in regular_expression_tokenizer.tokenize(sentence2):
    #     clean_train_reviews.append(word)

    # for word in regular_expression_tokenizer.tokenize(sentence3):
    #     clean_train_reviews.append(word)
    return clean_train_reviews


def create_bag_of_words(BoW_option, transform_option, stop):
    # word_list = create_test_list()
    # print(len(word_list))
    if transform_option == Transform.none:
        doc_list = pickle.load(open('processed_data_transform_none.pkl', 'rb'))
        sentiment_list = pickle.load(open('sentiments_transform_none.pkl', 'rb'))
    if transform_option == Transform.lemmas:
        doc_list = pickle.load(open('processed_data_transform_lemmas.pkl', 'rb'))
        sentiment_list = pickle.load(open('sentiments_transform_lemmas.pkl', 'rb'))
    print('vectorizing and creating feature vectors')
    if BoW_option == BoWType.frequency:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                     preprocessor=None, stop_words=stop)
                                     # max_features=5000)

        # dist = np.sum(train_data_features, axis=0)
        # for tag, count in zip(vocab, dist):
        #     print(count, tag)
    if BoW_option == BoWType.binary:
        vectorizer = HashingVectorizer(analyzer="word",
                                       tokenizer=None,
                                       preprocessor=None,
                                       stop_words=stop)
                                       # n_features=5000)

    if BoW_option == BoWType.tfidf:
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None,
                                     preprocessor=None,
                                     stop_words=stop,
                                     max_features=5000)
    train_data_features = vectorizer.fit_transform(doc_list)
    train_data_features = train_data_features.toarray()    
    # vocab = vectorizer.get_feature_names()

    # print(train_data_features)
    return train_data_features, sentiment_list

def return_ones(row):
    equals_one = row == 1

    r = np.array(range(len(equals_one)))
    keys = r[equals_one]
    # for key in keys:
    #     print(key)
    return keys

def output_features(feature_set, sentiments, BoW_option, Transform_option):
    print('creating output file')
    if BoW_option == BoWType.frequency:
        if Transform_option == Transform.none:
            output_file = "BoW_frequency_phrase_none.out"
        if Transform_option == Transform.lemmas:
            output_file = "BoW_frequency_phrase_lemmas.out"
    if BoW_option == BoWType.binary:
        if Transform_option == Transform.none:
            output_file = "BoW_binary_phrase_none.out"
        if Transform_option == Transform.lemmas:
            output_file = "BoW_binary_phrase_lemmas.out"
    if BoW_option == BoWType.tfidf:
        if Transform_option == Transform.none:
            output_file = "BoW_tfidf_phrase_none.out"
        if Transform_option == Transform.lemmas:
            output_file = "BoW_tfidf_phrase_lemmas.out"
    with open(output_file, 'aw') as output:
        print('begin writing to file')
        features = np.array(feature_set)
        num_docs = features.shape[0]
        index = 0 # for sentiment list
        for i in range(num_docs):
            row = features[i][:]
            keys = return_ones(row)
            senti = sentiments[index]
            line_string = str(senti)+'\t'
            for key in keys:
                line_string += str(key)+':1'
                if key != keys[len(keys)-1]:
                    line_string += ', '
            output.write(line_string+'\n')
            if (i % 50 == 0):
                print(float((i / float(num_docs))) * 100, " % complete")
            index += 1
        
feature_set, sentiments = create_bag_of_words(BoWType.frequency, Transform.none, None)
print('feature vectors created')
output_features(feature_set, sentiments, BoWType.frequency, Transform.none)
# print(feature_set)
