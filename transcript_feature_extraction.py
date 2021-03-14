import pandas as pd
import numpy as np
import nltk
from nrclex import NRCLex
import string


# CITATION: Some of these function are taken from my own (Charles Reinertson) EECS 445 Project 1. 

def getFeatures(data, features):
    """
    Create a numpy array with shape (num_samples, num_features) for input data. Each row contains
    the number of the percentage of each feature based off the total number of tokens in the word.
    Input:
        data: dataframe that has a transcript and category column label
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of features)
    """
    data["text"] = data["text"].map(lambda x: x.lower()).copy()
    dataframe = np.zeros((data.shape[0], len(features)))
    emotion_text_column = data["text"].map(lambda x: NRCLex(x)).copy()# NRCLex(data["text"])
    
   
    for i in range(dataframe.shape[0]):
        tokens = nltk.word_tokenize(data['text'].iloc[i])
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

def extract_dictionary(df):
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
    text = df["text"].map(lambda x: x.lower()).copy()
    k = 0
    
    for i in text:
        for j in string.punctuation:
            i = i.replace(j, ' ')
        words = i.split()
        for j in words:
            if j not in word_dict:
                word_dict[j] = k
                k += 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a bag of words count for each review.
    Use the word_dict to find the correct index to increment by 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has a transcript and category column label
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    # get the text in a vector
    text = df["text"].map(lambda x: x.lower()).copy()

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












def main():
    amazon = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt', delimiter = "\t", quoting=3, header=None)
    amazon.columns = ['text', 'label']


    features = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'sadness': 5,
            'surprise': 6, 'trust': 7, 'negative': 8, 'positive': 9, 'CC': 10, 'IN': 11, 'JJR': 12,
            'JJS': 13, 'PRP': 14}


    # getting the feature representation of each transcript in the above format
    transcript_features1 = getFeatures(amazon, features)

    # bag of words
    word_dict = extract_dictionary(amazon)
    transcript_features2 = generate_feature_matrix(amazon, word_dict)


    # combine bag of words and emotional feature representation. We can do this becuase both are normalized
    transcript_features = np.hstack((transcript_features1, transcript_features2))


 


if __name__ == "__main__":
    main()
