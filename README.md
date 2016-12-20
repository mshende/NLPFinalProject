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


David: 
- BagOfSentimentsTreeBuilder.py: Program for creating a sentiment tree (bag-of-sentiments).
     For each word in the word list we use WordNet to obtain all of the senses for the word. 
     This is done by getting the sets of synonyms called synsets for the word. 
     Each sense of the word in the synset is then entered into sentiwordnet senti_synset
     to obtain the pos_score, neg_score, and obj_score of each word sense.

    FORMAT OF OUTPUT: (BOP (U *) (U *) (U *) (U *) (P *) (P *) (U *) (U *) (U *) (P *) (U *) (P *) (U *) (U *) (U *))
    
- SentimentTreeBuilder.py: Program for creating a sentiment tree (bag-of-sentiments).
    The positive words are replaced in the parse tree representation of the sentence with a "P".
    The negative words are replaced in the parse tree representation of the sentence with a "N".
    The neutral words are replaced in the parse tree representation of the sentence with a "U".

    FORMAT OF OUTPUT: (ROOT (S (VP (VB U) (NP (NP (NN U)) (CC U) (NP (DT U) (JJ U) (NN U)))) (. .)))
    
- BagOfWords.py: Example BagOfWords program.

- bos_tree_files: The sentiment tree file outputs from BagOfSentimentsTreeBuilder.py and SentimentTreeBuilder.py
    to be used by the Tree Kernel
    
- SentenceVectorsClassifierExample.java: Implementation of the Doc2Vec paragraph vector that will provide a fine grained 
    sentiment analysis (negative, somewhat negative, neutral, somewhat positive, positive)
    
- SentenceTwoVectorsClassifierExample.java: Implementation of the Doc2Vec paragraph vector that will provide a course grained
    sentiment analysis (positive, negative)
    