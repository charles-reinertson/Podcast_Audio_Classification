import pandas as pd
import numpy as np
import nltk
import torch.cuda
from nrclex import NRCLex
import string
import enchant
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, DistilBertModel, DistilBertTokenizerFast


# CITATION: Some of these function are taken from my own (Charles Reinertson) EECS 445 Project 1. 

class BertEncoder(object):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, encoding_type):
        if encoding_type == 'distilbert':
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.encoding_fn = self._get_encoding_fn(encoding_type)

    def __call__(self, inputs, *args, **kwargs):
        inputs = self.tokenizer(inputs, return_tensors='pt', padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            return self.encoding_fn(inputs)

    def _get_encoding_fn(self, encoding_type):
        if encoding_type == 'bert_pooled':
            return lambda inputs: self.model(**inputs).pooler_output
        else:
            return lambda inputs: self.model(**inputs).last_hidden_state[:, 0, :]


def remove_non_english(transcripts):
    dictionary = enchant.Dict("en_US")
    validated_transcripts = []
    for transcript in transcripts:
        valid_tokens = []
        for token in transcript.split():
            if dictionary.check(token):
                valid_tokens.append(token)
        validated_transcripts.append(' '.join(valid_tokens))
    return validated_transcripts


def bert_encode_transcripts(transcripts, encoding_type='bert', num_split=5000):
    transcripts = remove_non_english(transcripts)
    numpy_transcripts = np.array(transcripts)
    encoder = BertEncoder(encoding_type)
    cls_encoding = []
    for numpy_transcripts in tqdm(np.array_split(numpy_transcripts, num_split)):
        cls_encoding.append(encoder(numpy_transcripts.tolist()))
    return torch.vstack(cls_encoding).cpu().numpy()


def get_emotion_features(series):
    """
    Create a numpy array with shape (num_samples, num_features) for input data. Each row contains
    the number of the percentage of each feature based off the total number of tokens in the word.
    Input:
        data: dataframe that has a transcript and category column label
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of features)
    """
    features = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'sadness': 5,
                'surprise': 6, 'trust': 7, 'negative': 8, 'positive': 9, 'CC': 10, 'IN': 11, 'JJR': 12,
                'JJS': 13, 'PRP': 14}
    series = series.map(lambda x: x.lower()).copy()
    dataframe = np.zeros((series.shape[0], len(features)))
    emotion_text_column = series.map(lambda x: NRCLex(x)).copy()# NRCLex(data)
    
   
    for i in range(dataframe.shape[0]):
        tokens = nltk.word_tokenize(series.iloc[i])
        parts_of_speach = nltk.pos_tag(tokens)
        
        key = list(emotion_text_column[i].raw_emotion_scores.keys())
        dict_emotion_scores = emotion_text_column[i].raw_emotion_scores
        for j, value in enumerate(key):
            dataframe[i, features[value]] += dict_emotion_scores[value]
            
        for j, value in enumerate(parts_of_speach):
            if value[1] in features.keys():
                dataframe[i, features[value[1]]] += 1

        dataframe[i] = dataframe[i] / len(tokens)
        
    return dataframe


def extract_dictionary(transcript_series):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    # check if the word is a english word or a bad trasncription
    d = enchant.Dict("en_US")
    text = transcript_series.map(lambda x: x.lower()).copy()
    k = 0
    
    for i in text:
        for j in string.punctuation:
            i = i.replace(j, ' ')
        words = i.split()
        for j in words:
            # only put word in dictionary if transcription is correct and word not in dictionary already
            if (j not in word_dict) and d.check(j):
                word_dict[j] = k
                k += 1
    return word_dict


def get_bag_of_words(series, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a bag of words count for each review.
    Use the word_dict to find the correct index to increment by 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        series: series that has a list of transcripts
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = series.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    # get the text in a vector
    text = series.map(lambda x: x.lower()).copy()

    k = 0
    for i in text:
        for j in string.punctuation:
            i = i.replace(j, ' ')
        words = i.split()
        for j in words:
            if j in word_dict:
                feature_matrix[k][word_dict[j]] += 1
        feature_matrix[k] /= len(words)
        k += 1
    return feature_matrix


def process_transcripts(data, combined_data, feature_set={'bag_of_words'}):
    '''
    data: either a train, test, or validate dataframe
    combined_data: a combination of train, test and validate
    feature_set: the type of feature representations that the transcripts will be converted to
    '''
    features = []
    if 'emotion' in feature_set:
        # getting the feature representation of each transcript in the above format
        features.append(get_emotion_features(data))

    if 'bag_of_words' in feature_set:
        # bag of words
        word_dict = extract_dictionary(combined_data)
        features.append(get_bag_of_words(data, word_dict))

    if 'bert_pooled' in feature_set:
        features.append(bert_encode_transcripts(data,
                                                encoding_type='bert_pooled'))

    if 'bert' in feature_set:
        features.append(bert_encode_transcripts(data,
                                                encoding_type='bert'))

    if 'distilbert' in feature_set:
        features.append(bert_encode_transcripts(data,
                                                encoding_type='distilbert'))


    # combine features
    return np.hstack(features)


def main():
    amazon = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt', delimiter = "\t", quoting=3, header=None)
    amazon.columns = ['text', 'label']

    process_transcripts(amazon.text, amazon.text)


if __name__ == "__main__":
    main()



