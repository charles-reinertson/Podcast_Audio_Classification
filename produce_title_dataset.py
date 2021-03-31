"""
######################### NOTE ################################
##### This file is intended to be run on Google Colab     #####
##### which is reflected in the read/write file structure #####
###############################################################
"""
import pickle
import pandas as pd
from tqdm.notebook import trange


ROOT_DIR = '/content/drive/MyDrive/Podcast_Audio_Classification'


def produce_titles():
    partition_stops = {'test': 73,
                       'train1': 91,
                       'train2': 88,
                       'train3': 81,
                       'validate': 66}

    for partition in partition_stops.keys():
        df = pd.read_csv(f'{ROOT_DIR}/{partition}.csv')
        for batch_num in trange(partition_stops[partition], desc=partition):
            with open('{}/data/{}_{}_failed.pkl'.format(ROOT_DIR, partition, batch_num),
                      mode='rb') as file:
                failed_in_batch = pickle.load(file)
            start_index = batch_num * 100
            stop_index = start_index + 100
            df_batch = df[start_index:stop_index]

            # the highest pickle protocol that Colab supports is 4
            colab_pickle_protocol = 4
            with open('{}/data/{}_{}_titles.pkl'.format(ROOT_DIR, partition, batch_num),
                      mode='wb') as file:
                pickle.dump(df_batch.drop(failed_in_batch).title_right.to_list(),
                            file,
                            protocol=colab_pickle_protocol)


if __name__ == "__main__":
    produce_titles()
