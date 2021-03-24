"""
######################### NOTE ################################
##### This file is intended to be run on Google Colab     #####
##### which is reflected in the read/write file structure #####
###############################################################

Usage:
    !python audio_feature_extraction.py [dataset partition] [starting index] [starting batch number]
"""
import sys
import pandas as pd
import numpy as np
import pickle
from audio_processor import AudioProcessor


ROOT_DIR = '/content/drive/MyDrive/Podcast_Audio_Classification'


def extract_features():
    partition, start_index, batch_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    df = pd.read_csv('{}/{}.csv'.format(ROOT_DIR, partition))
    batch_size = 1000
    processor = AudioProcessor()
    while start_index < df.shape[0]:
        stop_index = min(start_index + batch_size, df.shape[0])
        audio_features, transcripts, categories, failed_indices = \
            processor.process(df[['audio', 'audio_length', 'categories']][start_index:stop_index])

        np.save('{}/{}_{}_audio_features.npy'.format(ROOT_DIR, partition, batch_num),
                audio_features)
        # the highest pickle protocol that Colab supports is 4
        colab_pickle_protocol = 4
        with open('{}/{}_{}_transcripts.pkl'.format(ROOT_DIR, partition, batch_num),
                  mode='wb') as file:
            pickle.dump(transcripts, file, protocol=colab_pickle_protocol)
        with open('{}/{}_{}_labels.pkl'.format(ROOT_DIR, partition, batch_num),
                  mode='wb') as file:
            pickle.dump(categories, file, protocol=colab_pickle_protocol)
        with open('{}/{}_{}_failed.pkl'.format(ROOT_DIR, partition, batch_num),
                  mode='wb') as file:
            pickle.dump(failed_indices, file, protocol=colab_pickle_protocol)

        print('Stop index (excluded): {}'.format(stop_index))
        print('Num examples failed processing: {}'.format(len(failed_indices)))

        start_index = stop_index
        batch_num += 1


if __name__ == '__main__':
    extract_features()
