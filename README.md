# NLPFinalProject
Final Project for NLP Fall 2016

Maya:
- MayaBagOfWords.py: contains the code to generate the feature vectors for the Bag of Words models that we used. The various parameters that can be changed here are:
  1. BoWType: binary, frequency, or tfidf
  2. TransformType: none, lemmas, or pos
  3. stopwords: none or english
  4. number of features: integer value
  
- output_files/ : directory containing all of the output files of feature vectors. 36 files total, each with a different combination of parameters from the list above.
- process_data.py: use this to parse the .txt files from Eli's preprocessing stage and extract the sentences to be used in Bag of Words model; generates a list of sentences and a corresponding list of sentiments

Eli:

- utils.py: contains the code used to generate output files for the two datasets. The various formats and output files are:
  1. dataset_one_lemmas.txt: sentiment and lemmas for Rotten Tomatoes dataset
  2. dataset_one_pos_filtered.txt: sentiment and words (only nouns, adverbs, adjectives, and verbs) for Rotten Tomatoes dataset
  3. dataset_one_sentences.txt: sentiment and words for Rotten Tomatoes dataset


