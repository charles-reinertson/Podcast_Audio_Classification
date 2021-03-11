import logging
import opensmile
import audiofile
import pandas as pd
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from vosk import Model, KaldiRecognizer, SetLogLevel
from audio_feature_extraction import AudioProcessor


def test_vosk(url):
    # init values from test_ffmpeg.py example file in vosk repository
    sample_rate = 16000
    print('create model')
    model = Model("model")
    print('create recognizer')
    recognizer = KaldiRecognizer(model, sample_rate)

    # temporary file for storing of byte stream
    tmpfile = 'tmp_audio.wav'

    # run ffmpeg to process audio (can use url)
    subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-t', '60',  '-i',
                    url,
                    '-ar', str(sample_rate), '-ac', '1', tmpfile])

    # read byte stream from file
    # note:  can also read from return of subprocess.run() output is piped to stdout
    data = open(tmpfile, 'rb').read()
    # recognize speech
    print('recognize')
    recognizer.AcceptWaveform(data)
    # get 'text' field from json-formatted string FinalResult
    print('get result')
    text = json.loads(recognizer.FinalResult())["text"]
    print(text)
    # delete temporary file for byte stream
    subprocess.run(['rm', tmpfile])


def test_opensmile():
    files = ["data/fa36a26a1879453f95da1379c737cd6d_audio.wav"] * 2
    # signal, sampling_rate = audiofile.read(file, always_2d=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=2
    )
    df = smile.process_files(files, ends=[pd.to_timedelta('60s')] * 2)

    print(df)


def visualize_features():
    df_episodes = pd.read_csv('data/episodes.csv')
    audio_processor = AudioProcessor()
    audio_feature_data = []
    for index, row in df_episodes[3:7].iterrows():
        try:
            features, transcript = audio_processor.process(row.audio,
                                                           row.audio_length)
            audio_feature_data.append(features)
            for transcript_i in transcript:
                print(transcript_i)
        except subprocess.CalledProcessError:
            logging.error('Failed to load url {}'.format(row.audio))

    audio_feature_data = pd.DataFrame(audio_feature_data)
    audio_feature_data.boxplot(rot=-45, showfliers=False, showbox=False)
    plt.show()


def main():
    """MAIN FUNCTION"""
    # test_opensmile()
    # test_vosk('http://95bfm.com/sites/default/files/291117_Dear_Science.mp3')
    visualize_features()


if __name__ == '__main__':
    main()