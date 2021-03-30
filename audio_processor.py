import opensmile
import subprocess
import deepspeech
import soundfile
import io
import pandas as pd
import numpy as np
from tqdm import tqdm


class AudioProcessor:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=1
    )

    model = deepspeech.Model('deepspeech/model.pbmm')

    def __int__(self):
        self.model.enableExternalScorer('deepspeech/model.scorer')

    def process(self, df, sample_length=30):
        """
        Extract features from each url by downloading and sampling two
        (sample_length) seconds samples of the file and transcribing them;
        the audio features of the two samples are combined by averaging them

        :param df: a dataframe that specifies the location and duration
        of the audio file
        :param sample_length: the length of the audio samples used to extract features
        :return: features, transcripts, indices - an (n x k) dataframe, an n-length pandas series,
        an n-length list of the indices that were successfully processed
        """
        features, transcripts, titles, categories, failed_indices = ([],) * 5
        for index, (url, duration, title, category) in tqdm(df.iterrows(),
                                                            total=len(df),
                                                            desc='AudioProcessor'):
            if duration < sample_length:
                failed_indices.append(index)
                continue

            starts = get_random_start_times(duration, sample_length)
            samples = get_audio_samples(url, self.model.sampleRate(), starts,
                                        sample_length)
            if samples is None:
                failed_indices.append(index)
                continue

            samples_features = self.extract_audio_features(samples)
            samples_transcript = self.transcribe(samples)
            features.append(samples_features)
            transcripts.append(samples_transcript)
            titles.append(title)
            categories.append(category)

        if len(features) == 0:
            return [None] * 4
        return np.vstack(features), transcripts, titles, categories, failed_indices

    def transcribe(self, audio_signals):
        """
        Transcribe the given audio files;  uses the deepspeech library for transcription

        :param audio_signals: list of two signals to transcribe
        :return: combined transcript of both audio signals
        """
        stream = self.model.createStream()
        stream.feedAudioContent(audio_signals[0])
        stream.feedAudioContent(audio_signals[1])
        return stream.finishStream()

    def extract_audio_features(self, audio_signals):
        """
        Analyze the given audio file for acoustic features determined by the
        configuration of the opensmile instance: self.smile

        :param audio_signals: the signals from which to extract features
        :returns: 1xn pandas dataframe which includes labels for each feature
        """
        sample_rate = self.model.sampleRate()
        df1 = self.smile.process_signal(audio_signals[0], sample_rate)
        df2 = self.smile.process_signal(audio_signals[1], sample_rate)
        averaged_features = pd.concat([df1, df2]).mean(axis=0)
        return averaged_features.to_numpy()


def get_random_start_times(audio_length, sample_length):
    max_start_time = audio_length - sample_length
    rng = np.random.default_rng()
    starts = np.sort(rng.integers(max_start_time, size=2, endpoint=True))
    starts = separate_pair_by_interval(starts, max_start_time, sample_length)
    return starts


def separate_pair_by_interval(sorted_pair, max_value, interval):
    low, high = sorted_pair.copy()
    if high < low + interval:
        high = min(max_value, low + interval)
        if high < low + interval:
            low = max(0, high - interval)

    return low, high


def get_audio_samples(url, sample_rate, start_times, sample_length):
    samples = []
    for start in start_times:
        process = subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-ss',
                                  str(start), '-t', str(sample_length), '-i',
                                  url, '-ar', str(sample_rate), '-ac', '1',
                                  '-f', 'wav', '-'],
                                 stdout=subprocess.PIPE)
        if process.returncode != 0:
            return None
        audio_signal, _ = soundfile.read(io.BytesIO(process.stdout), dtype='int16')
        samples.append(audio_signal)
    return samples
