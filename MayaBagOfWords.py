#Author: Maya Shende
from __future__ import print_function

from enum import Enum
import pandas as pd
import numpy as np
from MayaUtils import Transform
from MayaUtils import process_file
from sklearn.feature_extraction.text import CountVectorizer
import math

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


def create_words_list():
    output_tokens = process_file("train.tsv", Transform.tokens)
    word_list = []
    for item in output_tokens:
        words = item[1]
        for word in words:
            if word not in word_list:
                word_list.append(word)
    return word_list


def create_bag_of_words():
    word_list = create_words_list()
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words=None,
                                 max_features=5000)
    train_data_features = vectorizer.fit_transform(word_list)
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()
    print("vocab: ", vocab)


# def create_bag_of_words(input_file):
#     words = []
#     term_frequency = {}
#     inverse_doc_frequency = {}
#     print('beginning processing file')
#     output_tokens = process_file("train.tsv", Transform.tokens)
#     print('finished processing file')
#     doc_number = 0
#     print('begin iterating over phrases')
#     for item in output_tokens:
#         word_list = item[1]
#         word_freq = {}
#         for word in word_list:
#             if word in inverse_doc_frequency:
#                 inverse_doc_frequency[word] += 1
#             else:
#                 inverse_doc_frequency[word] = 1
#             if word in word_freq:
#                 word_freq[word] += 1
#             else:
#                 word_freq[word] = 1
#             if word not in words:
#                 words.append(word)
#         term_frequency[doc_number] = word_freq
#         doc_number += 1
#     print('created list of words and dictionaries')
            
#     print('initializing term-doc-matrix')
#     num_words = len(words)
#     num_phrases = len(output_tokens)
#     term_doc = [[0 for i in range(num_words)] for j in range(num_phrases)]
    
#     print('begin filling in matrix')
#     for document in range(num_phrases):
#         for word in range(num_words):
#             tf_idf = compute_tfidf(words[word], term_frequency,
#                                    document, num_phrases, inverse_doc_frequency)
#             term_doc[document][word] = tf_idf
#     bag_of_words = pd.DataFrame(term_doc, columns=words)
#     print(bag_of_words)

    # with open(input_file, 'r') as training:
    #     original_file = pd.read_csv("train.tsv", sep='\t')
    #     print(original_file.iloc[0,2])
    #     training_lines = training.readlines()
    #     word_dictionary = {}
    #     index = 0
    #     for line in training_lines:
    #         phrase = original_file.iloc[index,0]
    #         if line != '':
    #             line = line.strip().split(',')
    #             for i in range(1, len(line)):
    #                 if line[i] in word_dictionary:
    #                     word_dictionary[line[i]].append(phrase)
    #                 else:
    #                     word_dictionary[line[i]] = [phrase]
    #         index+=1
    #     print(word_dictionary)
    

create_bag_of_words()
