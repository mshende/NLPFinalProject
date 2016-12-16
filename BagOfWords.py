sentence1 = 'This is a review sentence'
sentence2 = 'This is another review sentence'
sentence3 = 'Test of the bag of words'
from nltk.tokenize import RegexpTokenizer
regular_expression_tokenizer = RegexpTokenizer(r'\w+')

clean_train_reviews = []

for word in regular_expression_tokenizer.tokenize(sentence1):
    clean_train_reviews.append(word)

for word in regular_expression_tokenizer.tokenize(sentence2):
    clean_train_reviews.append(word)

for word in regular_expression_tokenizer.tokenize(sentence3):
    clean_train_reviews.append(word)


print(clean_train_reviews)

print("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()


print(train_data_features.shape)


# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)