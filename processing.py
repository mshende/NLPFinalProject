import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet


def is_valid_pos(data):
    if data[0] == 'N' or data[0] == 'V' or data[0] == 'R' or data[0] == 'J':
        return True
    else:
        return False


def convert_sentiment(original):
    if original == 'positive':
        return 4
    elif original == 'neutral':
        return 2
    elif original == 'negative':
        return 0


def wn_pos_mapper(tag):
        # POS mapper to convert the nltk POS tag into the wordnet POS tag
        # This function is necessary because the nltk pos_tag method is used
        # for the pos tagging while WordNetLemmatizer is used for the lemmatization.
        # Therefore the nltk POS tag must be mapped to the WordNet POS tag which will
        # be used as the input for the WordNetLemmatizer

        # Input arguments are:
        #   tag: the POS tag that is output from the nltk.pos_tag method
        # Output:
        #   WordNet POS tag mapped from the nltk POS tag

        # Example:
        # Input tag: N
        # Output tag: wordnet.NOUN

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('S'):
        return wordnet.ADJ_SAT
    else:
        return ''


# Open files
file1 = pd.read_csv("train.tsv", sep='\t')
file2 = pd.read_csv("airline_tweets.csv")

# First, deal with file1

# Go through data set and only pull out the full sentences
# sentences = []
# current_sent_id = 0
# for entry_index in range(len(file1)):
#     entry = file1.iloc[entry_index, :]
#     # Check the sent_id against the current_sent_id
#     if entry["SentenceId"] > current_sent_id:
#         current_sent_id = entry["SentenceId"]
#         sentences.append((entry["Sentiment"], entry["Phrase"]))


# Now, print out different formats to output files

# # Original sentences
# file1_output_sentences = open("dataset_one_sentences.txt", "w+")
# for sentiment, text in sentences:
#     file1_output_sentences.write(str(sentiment) + "\t" + text + "\n")


# # POS filtered
# file1_output_pos_filtered = open("dataset_one_pos_filtered.txt", "w+")
# for sentiment, text in sentences:
#     tokens = word_tokenize(text)
#     tags = pos_tag(tokens)
#     final_text = ""
#     for word, tag in tags:
#         if is_valid_pos(tag):
#             final_text += word + " "
#     file1_output_pos_filtered.write(str(sentiment) + "\t" + final_text + "\n")


# # Lemmatized Sentences
# lemmatizer = WordNetLemmatizer()
# file1_output_lemmas = open("dataset_one_lemmas.txt", "w+")
# for sentiment, text in sentences:
#     tokens = word_tokenize(text)
#     tags = pos_tag(tokens)
#     final_text = ""
#     for word, tag in tags:
#         pos = wn_pos_mapper(tag)
#         if pos != '':
#             final_text += lemmatizer.lemmatize(word, pos) + " "
#         else:
#             final_text += lemmatizer.lemmatize(word) + " "
#     file1_output_lemmas.write(str(sentiment) + "\t" + final_text + "\n")


# Now, go through second file
tweets = []
for entry_index in range(len(file2)):
    entry = file2.iloc[entry_index, :]
    tweets.append((convert_sentiment(entry["airline_sentiment"]), entry["text"]))

# file2_output_sentences = open("dataset_two_sentences.txt", "w+")
# for sentiment, text in tweets:
#     file2_output_sentences.write(str(sentiment) + "\t" + str(text.encode('ascii', 'ignore')) + "\n")


# # POS filtered
# file2_output_pos_filtered = open("dataset_two_pos_filtered.txt", "w+")
# tokenizer = TweetTokenizer()
# for sentiment, text in tweets:
#     tokens = tokenizer.tokenize(text)
#     tags = pos_tag(tokens)
#     final_text = ""
#     for word, tag in tags:
#         if is_valid_pos(tag):
#             final_text += word + " "
#     file2_output_pos_filtered.write(str(sentiment) + "\t" + str(final_text.encode('ascii', 'ignore')) + "\n")

# Lemmatized Sentences
lemmatizer = WordNetLemmatizer()
file2_output_lemmas = open("dataset_two_lemmas.txt", "w+")
tokenizer = TweetTokenizer()
for sentiment, text in tweets:
    tokens = tokenizer.tokenize(text)
    tags = pos_tag(tokens)
    final_text = ""
    for word, tag in tags:
        pos = wn_pos_mapper(tag)
        if pos != '':
            final_text += lemmatizer.lemmatize(word, pos) + " "
        else:
            final_text += lemmatizer.lemmatize(word) + " "
    file2_output_lemmas.write(str(sentiment) + "\t" + str(final_text.encode('ascii', 'ignore')) + "\n")
