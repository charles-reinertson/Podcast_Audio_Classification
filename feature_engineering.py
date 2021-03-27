import os
import numpy as np
import pandas as pd
import glob
import argparse
from transcript_feature_extraction import *
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_num_batches',
    type=int,
    default=6,
    help='The number of batches of test data')
parser.add_argument(
    '--train1_num_batches',
    type=int,
    default=16,
    help='The number of batches of train 1 data')
parser.add_argument(
    '--train2_num_batches',
    type=int,
    default=19,
    help='The number of batches of train 2 data')
parser.add_argument(
    '--train3_num_batches',
    type=int,
    default=6,
    help='The number of batches of train 3 data')
parser.add_argument(
    '--validate_num_batches',
    type=int,
    default=4,
    help='The number of batches of validate data')



def get_train_test_validat(data_location, test_num_batches, train1_num_batches, train2_num_batches, train3_num_batches, validate_num_batches):
    '''
    data_location: the filepath to load data
    test_num_batches: The number of batches of test data
    train_num_batches: The number of batches of train data
    validate_num_batches: The number of batches of validate data

    '''
    lst_test_audio_features = []
    lst_test_labels = []
    lst_test_transcripts = []

    # TEST DATA
    for i in range(test_num_batches):
        # read in test audio features, labels, and transcripts to list in order for future combine
        filepath_audio_features = data_location + 'test_{}_audio_features.npy'.format(i)
        filepath_labels = data_location + 'test_{}_labels.pkl'.format(i)
        filepath_transcripts = data_location + 'test_{}_transcripts.pkl'.format(i)

        lst_test_audio_features.append(np.load(filepath_audio_features))
        lst_test_labels.append(pd.read_pickle(filepath_labels))
        lst_test_transcripts.append(pd.read_pickle(filepath_transcripts))

    # combine list into one dataframe (must convert audio_features_test to pandas dataframe)
    audio_features_test = np.vstack(lst_test_audio_features)
    # combine labels into one list
    labels_test = [item for sublist in lst_test_labels for item in sublist]
    # combine transcripts into one list
    transcripts_test = [item for sublist in lst_test_transcripts for item in sublist]
    # convert to pandas array for audio features
    test = pd.DataFrame(audio_features_test)

    test['text'] = transcripts_test
    test['label'] = labels_test

    lst_train_audio_features = []
    lst_train_labels = []
    lst_train_transcripts = []

    # TRAIN DATA
    for j in range(1, 4):
        if (j == 1):
            num_batches = train1_num_batches
        elif (j == 2):
            num_batches = train2_num_batches
        else:
            num_batches = train3_num_batches
        for i in range(num_batches):
            # read in train audio features, labels, and transcripts to list in order for future combine
            filepath_audio_features = data_location + 'train{}_{}_audio_features.npy'.format(j, i)
            filepath_labels = data_location + 'train{}_{}_labels.pkl'.format(j, i)
            filepath_transcripts = data_location + 'train{}_{}_transcripts.pkl'.format(j, i)

            lst_train_audio_features.append(np.load(filepath_audio_features))
            lst_train_labels.append(pd.read_pickle(filepath_labels))
            lst_train_transcripts.append(pd.read_pickle(filepath_transcripts))

    # combine list into one dataframe (must convert audio_features_test to pandas dataframe)
    audio_features_train = np.vstack(lst_train_audio_features)
    # combine labels into one list
    labels_train = [item for sublist in lst_train_labels for item in sublist]
    # combine transcripts into one list
    transcripts_train = [item for sublist in lst_train_transcripts for item in sublist]
    # convert to pandas array for audio features
    train = pd.DataFrame(audio_features_train)

    train['text'] = transcripts_train
    train['label'] = labels_train


    lst_validate_audio_features = []
    lst_validate_labels = []
    lst_validate_transcripts = []

    # VALIDATE DATA
    for i in range(validate_num_batches):
        # read in validate audio features, labels, and transcripts to list in order for future combine
        filepath_audio_features = data_location + 'validate_{}_audio_features.npy'.format(i)
        filepath_labels = data_location + 'validate_{}_labels.pkl'.format(i)
        filepath_transcripts = data_location + 'validate_{}_transcripts.pkl'.format(i)

        lst_validate_audio_features.append(np.load(filepath_audio_features))
        lst_validate_labels.append(pd.read_pickle(filepath_labels))
        lst_validate_transcripts.append(pd.read_pickle(filepath_transcripts))

    # combine list into one dataframe (must convert audio_features_test to pandas dataframe)
    audio_features_validate = np.vstack(lst_validate_audio_features)
    # combine labels into one list
    labels_validate = [item for sublist in lst_validate_labels for item in sublist]
    # combine transcripts into one list
    transcripts_validate = [item for sublist in lst_validate_transcripts for item in sublist]
    # convert to pandas array for audio features
    validate = pd.DataFrame(audio_features_validate)

    validate['text'] = transcripts_validate
    validate['label'] = labels_validate
   
    
    


    return train, test, validate

def feature_engineer(args):
    
    # N rows, 90 columns with 88 of those columns being audio features (0-88 )and the last two columns being 'text' and label'
    train, test, validate = get_train_test_validat('./data/data/', test_num_batches=args.test_num_batches, 
    train1_num_batches=args.train1_num_batches, train2_num_batches=args.train2_num_batches, train3_num_batches=args.train3_num_batches,
    validate_num_batches=args.validate_num_batches)

    # combine train test and split to create a dictionary during bag of words with all features needed
    df_for_bag_of_words_dict = pd.concat([train.text, test.text, validate.text], axis=0)
    # turn transcript into combination of bag of words features and emotion/POS features
    train_transcript_features = process_transcripts(train.text, df_for_bag_of_words_dict)
    test_transcript_features = process_transcripts(test.text, df_for_bag_of_words_dict)
    validate_transcript_features = process_transcripts(validate.text, df_for_bag_of_words_dict)

    # drop column text because we now have a feature representation for the text
    train = train.drop(columns=['text'])
    test = test.drop(columns=['text'])
    validate = validate.drop(columns=['text'])

    # put labels in seperate dataframe
    train_labels = train.label
    test_labels = test.label
    validate_labels = validate.label

    # One hot encode labels
    train_labels = LabelBinarizer().fit_transform(train_labels)
    test_labels = LabelBinarizer().fit_transform(test_labels)
    validate_labels = LabelBinarizer().fit_transform(validate_labels)

    # drop labels from main dataframe
    train = train.drop(columns=['label'])
    test = test.drop(columns=['label'])
    validate = validate.drop(columns=['label'])

    # TODO: Normalize features in train, test, and validate (normalize speech features)

    # combine transcript feauters and speech features
    train_transcript_features = pd.concat([train, pd.DataFrame(train_transcript_features)], axis=1)
    test_transcript_features = pd.concat([test, pd.DataFrame(test_transcript_features)], axis=1)
    validate_transcript_features = pd.concat([validate, pd.DataFrame(validate_transcript_features)], axis=1)

    # convert features to numpy from pandas
    train_transcript_features = train_transcript_features.to_numpy()
    test_transcript_features = test_transcript_features.to_numpy()
    validate_transcript_features = validate_transcript_features.to_numpy()

    # create a final train, test, validate dataframe with all features correct and ready to be modeled
    train = pd.DataFrame({'categories': list(train_labels), 'transcript_features': list(train_transcript_features)}, columns=['categories', 'transcript_features'])
    print(train.head())

    test = pd.DataFrame({'categories': list(test_labels), 'transcript_features': list(test_transcript_features)}, columns=['categories', 'transcript_features'])
    print(test.head())

    validate = pd.DataFrame({'categories': list(validate_labels), 'transcript_features': list(validate_transcript_features)}, columns=['categories', 'transcript_features'])
    print(validate.head())
  

    # location of saved dataframes
    test_file_location = "./data/test_dataframe.pkl"
    train_file_location = "./data/train_dataframe.pkl"
    validate_file_location = "./data/validate_dataframe.pkl"

    # save dataframe to pkl file
    train.to_pickle(train_file_location)
    test.to_pickle(test_file_location)
    validate.to_pickle(validate_file_location)


    

 
    






def main(args):
    '''
    produce, on a rolling basis, processed batches of output from the dataset. I can run five compute instances of colab at once, 
    so I will be able to extract features from partitions train1, train2, train3, test, validate. All of these partitions will be 22000 examples.
    Note that the 66000 examples in train are split into 3 separate partitions just so that I can parallelize the compute. 
    For every batch, the partition will output 3 files:

    [name]_[batch num]_audio_features.npy
    [name]_[batch num]_labels.npy
    [name]_[batch num]_transcripts.npy

    Note that “name” is either train1, train2, train3, test, or validate depending on what partition the data come from and “batch_num” will 
    start at 0 and count up as I successfully process batches. Also note that the filenames will NOT include brackets. 
    Those are there just to indicate variables in the filenames. The audio features will NOT be normalized because I am only working with the current batch.
    My proposal is that you write the next stage in the feature extraction which includes:
    1. Combining the batches (note that this is a dynamic process because more and more batches will become available throughout our development; 
    I suggest programming it with a command line argument or some other easy parameterization so you don’t have to change the code)
    2. Normalizing the audio features
    3. Extracting features from the transcripts based on the current full collection of transcripts
    4. Combining the audio features and transcript features into one array

    '''


    feature_engineer(args)

 


if __name__ == "__main__":
    main(parser.parse_args())
