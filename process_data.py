import pickle
from MayaUtils import Transform
from MayaUtils import process_file

# uses Eli's process_file utility function to extract the phrase and
# sentiment from the data
def create_words_list(transform_option):
    print('creating output tokens list')
    word_list = []
    sentiment_list = []
    if transform_option == Transform.none:
        with open("dataset_one_sentences.txt", 'r') as sentences:
            lines = sentences.readlines()
            for line in lines:
                line = line.strip().split('\t')
                sentiment_list.append(line[0])
                if len(line) == 1:
                    print('length was one')
                    word_list.append("")
                else:
                    word_list.append(line[1])
        # output_tokens = process_file("train.tsv", transform_option)
        # for item in output_tokens:
        #     #     words = item[1]
        #     # for word in words:
        #     #     if word not in word_list:
        #     #         word_list.append(word)
        #     word_list.append(item[1])
        #     sentiment_list.append(item[0])
    if transform_option == Transform.lemmas:
        with open("dataset_one_lemmas.txt", 'r') as lemmas:
            lines = lemmas.readlines()
            for line in lines:
                line = line.strip().split('\t')
                sentiment_list.append(line[0])
                if len(line) == 1:
                    word_list.append("")
                else:
                    word_list.append(line[1])
    if transform_option == Transform.pos:
        with open("dataset_one_pos_filtered.txt", 'r') as pos:
            lines = pos.readlines()
            for line in lines:
                line = line.strip().split('\t')
                sentiment_list.append(line[0])
                if len(line) == 1:
                    word_list.append("")
                else:
                    word_list.append(line[1])
    return word_list, sentiment_list

if __name__ == "__main__":
    transform = Transform.pos
    if transform == Transform.none:
        doc_list, sentiment_list = create_words_list(Transform.none)
        pickle.dump(doc_list, open('processed_data_transform_none.pkl', 'wb'))
        pickle.dump(sentiment_list, open('sentiments_transform_none.pkl', 'wb'))
    if transform == Transform.lemmas:
        doc_list, sentiment_list = create_words_list(Transform.lemmas)
        pickle.dump(doc_list, open('processed_data_transform_lemmas.pkl', 'wb'))
        pickle.dump(sentiment_list, open('sentiments_transform_lemmas.pkl', 'wb'))
    if transform == Transform.pos:
        doc_list, sentiment_list = create_words_list(Transform.pos)
        pickle.dump(doc_list, open('processed_data_transform_pos.pkl', 'wb'))
        pickle.dump(sentiment_list, open('sentiments_transform_pos.pkl', 'wb'))
