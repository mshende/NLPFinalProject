# Author: Eli Andrew

from enum import Enum
from nltk.tokenize import word_tokenize
import pandas as pd


class Transform(Enum):
    # Transformation option to select output format of transformation.

    # This enum provides four options: none, tokens, lemmas, pos
    none = 0
    tokens = 1
    lemmas = 2
    pos = 3


def process_file(input_file, transform_option):
    # Pre-processing function to transform the input file into the desired format

    # Input arguments are:
    #   input_file: a tab separated file with the format: PhraseId  SentenceId  Phrase  Sentiment
    #   transform_option: Transform Enum that specifies the output format for the function

    # Output format:
    #   The output format is dependent on the transform_option that is provided. The transform_options
    #   along with their associated output formats are specified as follows:
    #       none: performs no transformation and output is an array of sentiment values for each line
    #       tokens: tokenizes each phrase and output is an array of tuples where each tuple is of the format
    #               (Sentiment, [tokens])
    #       lemmas: lemmatizes each phrase and output is an array of tuples where each tuple is of the format
    #               (Sentiment, [lemmas])
    #       pos: part of speech tags each phrase and the output is an array of tuples where each tuple is of
    #            the format (Sentiment, [(token, POS tag)])

    # Example:
    # input_file contents:
    #   11  1   demonstrating the adage 2
    #   12  1   demonstrating 2
    # output with transform_option = none
    #   [2, 2]
    # output with transform_option = tokens
    #   [(2, [demonstrating, the, adage]), (2, [demonstrating])]
    # output with transform_option = lemmas
    #   [(2, [demonstrate, the, adage]), (2, [demonstrate])]
    # output with transform_option = pos
    #   [(2, [(demonstrating, VBG), (the, DET), (adage, NN)]), (2, [(demonstrating, VBG)])]

    original_file = pd.read_csv(input_file, sep="\t")
    filtered_file = original_file.iloc[:, 2:5]

    processed_output = []

    for entry_index in range(len(filtered_file)):
        entry = filtered_file.iloc[entry_index, :]
        entry_phrase_tokens = word_tokenize(entry["Phrase"])
        processed_output.append((entry["Sentiment"], entry_phrase_tokens))

    if transform_option == Transform.none:
        return processed_output
    elif transform_option == Transform.tokens:
        return processed_output
    elif transform_option == Transform.lemmas:
        return processed_output
    elif transform_option == Transform.pos:
        return processed_output
    else:
        return []

process_file("train.tsv", Transform.none)
