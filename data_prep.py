import os
import numpy as np
import pandas as pd
import nltk
from collections import Counter
import random

def removeNonAplhabet(inputlist):
    # removes non alphabetical words in a list
    # inputlist: list that we want non alphabetical words removed
    return [w for w in inputlist if w.isalpha()]

def removeDuplicatesList(inputlist):
    # removes duplicates in a list
    # inputlist: list that we want duplicates removed
    res = []
    [res.append(x) for x in inputlist if x not in res]
    return res
 
    

def main():
    # The main function
    episodes_df = pd.read_csv('data/episodes.csv')
    podcasts_df = pd.read_csv('data/podcasts.csv')

    # join dataframes on unique column 'uuid' for podcasts and 'podcast_uuid' for episodes
    df = episodes_df.join(podcasts_df.set_index('uuid'), how='inner', on='podcast_uuid', lsuffix='_left', rsuffix='_right')
    # drop columns we do not care about
    df = df.drop(['pub_date', 'image', 'website', 'itunes_id'], axis=1)

    # drop rows where the language is not English
    df.drop(df.loc[df['language']!='English'].index, inplace=True)

    # delete later on just for better runtime
    # df = df.head(1000)
    

    print("----Dataframe Random Feature Vectors----")
    print(df.sample(3).T)
   
    
    print("---Count Missing Values Per Feature---")
    print(df.isna().sum())

    # drop rows with missing title
    # drop rows with missing audio
    df = df.dropna(subset=['title_left', 'title_right', 'audio'])

    # get each unique category
    category_labels = df['categories'].value_counts()
    print(len(category_labels))
    print(category_labels)


    # next part the goal is to simplify the categories column of df
    tokens = []
    items = {}
    print(df['categories'].head(100))
    df["categories"] = df["categories"].map(lambda x: x.lower()).copy()


    for i in range(df.shape[0]):
        tokens.append(nltk.word_tokenize(df['categories'].iloc[i]))
        tokens[i] = removeNonAplhabet(tokens[i])
        tokens[i] = removeDuplicatesList(tokens[i])
        
    # put list in dictionary to get most frequent words
    # fuction to flatten list
    flattened = lambda t: [item for sublist in t for item in sublist]
    # flattened tokens list
    tokens_flattened = flattened(tokens)
    # get most frequent items in list
    items = Counter(tokens_flattened)
    # only get the 20 most common podcast categories that we will train our data on
    items = dict(items.most_common(20))
    # remove news, religion, christianity, culture, recreation, arts, professional because they are repetitive
    ignore = ['news','religion','christianity','culture','recreation','arts','professional']
    for word in ignore:
        if word in items:
            del items[word]

    
    # simplify the category column name then convert list to string
    for index, token in enumerate(tokens):
        holder_token = []
        for i, value in enumerate(token):
            if value in items:
                holder_token.append(value)
        # only keep the feature vector if it has 1 categorical variable after ignoring repetitive words. This reduces the number of rows
        # and will make sure there is no ambigious categories
        if len(holder_token) != 1:
            tokens[index] = np.nan
        else:
            tokens[index] = holder_token[0]
        
        
    

    # assign list to existing column of Pandas Data Frame
    df['categories'] = tokens
    # get each unique category
    category_labels = df['categories'].value_counts()
    print('---Length of "Category" Column---')
    print(len(category_labels))
    print('---Categories of "Category" Column---')
    print(category_labels)

    # drop NaN in categories. Created because some features vectors don't have a category in our 13 categories chosen
    print(df.isna().sum())
    df = df.dropna(subset=['categories'])

    # get names of indexes for which 
    # column 'categories' has each value and drop these rows until there's 10,000 samples of each category 
    for index, value in enumerate(category_labels.index):
        index_names = (df[ df['categories'] == value ].index).values
        
        index_to_drop = random.sample(list(index_names), (index_names.shape[0] - 10000))
        
        # drop these row indexes 
        # from dataFrame 
        df.drop(index_to_drop, inplace = True) 

    # reset index
    df = df.copy().reset_index(drop=True)

    # get each unique category
    category_labels = df['categories'].value_counts()
    print('---Length of "Category" Column---')
    print(len(category_labels))
    print('---Categories of "Category" Column---')
    print(category_labels)



    
    


if __name__ == "__main__":
    main()
