import opensmile
import subprocess
import json
import numpy as np
import pandas as pd
from vosk import KaldiRecognizer, Model
import logging
from tqdm import tqdm


class AudioProcessor:
    """
    Class for audio processing;  instantiates language models and audio processing
    engines necessary for extracting features from audio
    """
    en_language_model = Model('model')
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=1
    )

    def process(self, urls_and_durations, language='en', sample_rate=16000):
        """
        extracts features from each url by downloading and sampling two 60 second clips
        of the file and combining those features by averaging the audio features
        and concatenating the transcripts

        :param urls_and_durations: a dataframe of urls that point to audio files and its duration
        :param language: the language spoken in all of the audio files
        :param sample_rate: the sample rate at which the audio files will be processed
        :return: features, transcripts, indecies - an (n x k) numpy array, an n-length pandas series,
        an n-length list of the indecies that were successfully processed
        """
        features, transcripts, indecies = [], [], []
        for index, (url, duration) in tqdm(urls_and_durations.iterrows()):
            try:
                starts = get_random_start_times(duration)
                clips = save_audio_clips(url, sample_rate, starts)
                url_features = self.extract_audio_features(clips)
                url_transcript = self.transcribe(clips, sample_rate, language)
                delete_files(clips)
                features.append(url_features)
                transcripts.append(url_transcript)
                indecies.append(index)
            except subprocess.CalledProcessError:
                logging.error('Failed to load url {}'.format(url))
            except AudioTooShort:
                logging.error('url {} is only {}s -- too short to analyze'.format(url, duration))
        return pd.DataFrame(features), pd.Series(transcripts), indecies

    def transcribe(self, audio_files, sample_rate, language):
        """
        transcribe the given audio files;  uses the vosk library for transcription
        :param audio_files: audio to transcribe
        :param sample_rate: the sample rate of all the audio files; must be same for all files
        :param language: the language spoken in all of the audio files
        :return: transcript of audio files
        """
        if language == 'en':
            speech_recognizer = KaldiRecognizer(self.en_language_model,
                                                sample_rate)
        else:
            return None

        transcript = ''
        for audio_file in audio_files:
            data = open(audio_file, 'rb').read()
            speech_recognizer.AcceptWaveform(data)
            text = json.loads(speech_recognizer.FinalResult())["text"]
            transcript += (' ' + text)
        return transcript

    def extract_audio_features(self, audio_files):
        """
        Anaylze the given audio file for low-level acoustic features
        returned in a pandas dataframe
        :param audio_files: the path to the audio file
        :returns: 1xn pandas dataframe which includes labels for each feature
        """
        df = self.smile.process_files(audio_files)
        return df.mean(axis=0)

class AudioTooShort(Exception):
    """ Raised when audio file is shorter than interval needed to analyze"""
    pass


def delete_files(filenames):
    for file in filenames:
        subprocess.run(['rm', file])


def get_random_start_times(duration, clip_length=60):
    max_start_time = duration - clip_length
    if max_start_time < 0:
        raise AudioTooShort
    rng = np.random.default_rng()
    starts = np.sort(rng.integers(max_start_time, size=2, endpoint=True))
    starts = separate_pair_by_interval(starts, max_start_time, clip_length)
    return starts


def separate_pair_by_interval(sorted_pair, max_value, interval):
    low, high = sorted_pair.copy()
    if high < low + interval:
        high = min(max_value, low + interval)
        if high < low + interval:
            low = max(0, high - interval)

    return low, high


def save_audio_clips(url, sample_rate, starts, interval=60):
    clip_filenames = ['temp_clip_1.wav', 'temp_clip_2.wav']
    for start, destination_filename in zip(starts, clip_filenames):
        subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-ss', str(start),
                        '-t', str(interval), '-i', url, '-ar', str(sample_rate),
                        '-ac', '1', destination_filename],
                       check=True)
    return clip_filenames
