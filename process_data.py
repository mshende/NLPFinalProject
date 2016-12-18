import pickle
from MayaUtils import Transform
from MayaUtils import process_file

# uses Eli's process_file utility function to extract the phrase and
# sentiment from the data
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

# doc_list: list of phrases (no lemmatization)
# sentiment_list: list of seniments
doc_list, sentiment_list = create_words_list(Transform.none)

# # doc_list: list of phrases (lemmatized)
# # sentiment_list: list of sentiments
# doc_list, sentiment_list = create_words_list(Transform.lemmas)

# creates pickled file of phrases to be used in MayaBagOfWords.py
pickle.dump(doc_list, open('processed_data_transform_none.pkl', 'wb'))
pickle.dump(doc_list, open('processed_data_transform_lemmmas.pkl', 'wb'))

#creates pickles file of sentiments to be used in MayaBagOfWords.py
pickle.dump(sentiment_list, open('sentiments_transform_none.pkl', 'wb'))
pickle.dump(sentiment_list, open('sentiments_transform_lemmas.pkl', 'wb'))
