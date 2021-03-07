import opensmile
import subprocess
import json
import numpy as np
from vosk import KaldiRecognizer, Model


class AudioProcessor:
    _en_language_model = Model('model')
    _smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1
    )

    def process(self, url, duration, language='en', sample_rate=16000):
        starts = get_random_start_times(duration)
        clips = save_audio_clips(url, sample_rate, starts, interval=60)
        features = self.extract_audio_features(clips)
        transcript = self.transcribe(clips, sample_rate, language)
        delete_files(clips)
        return features, transcript

    def transcribe(self, audio_files, sample_rate, language):
        if language == 'en':
            speech_recognizer = KaldiRecognizer(self._en_language_model,
                                                sample_rate)
        else:
            return None

        transcripts = []
        for audio_file in audio_files:
            data = open(audio_file, 'rb').read()
            speech_recognizer.AcceptWaveform(data)
            text = json.loads(speech_recognizer.FinalResult())["text"]
            transcripts.append(text)
        return transcripts

    def extract_audio_features(self, audio_files):
        """
        Anaylze the given audio file for low-level acoustic features
        returned in a pandas dataframe
        params: audio_file - the path to the audio file
        returns: 1xn pandas dataframe which includes labels for each feature
        """
        df = self._smile.process_files(audio_files)
        return df.mean(axis=0)


def save_audio_clip(url, sample_rate, destination_filename, start=0, interval=60):
    subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-ss', str(start),
                    '-t', str(interval), '-i', url, '-ar', str(sample_rate),
                    '-ac', '1', destination_filename],
                   check=True)
    return destination_filename


def delete_files(filenames):
    for file in filenames:
        subprocess.run(['rm', file])


def get_random_start_times(duration, clip_length=60):
    max_start_time = duration - clip_length
    rng = np.random.default_rng()
    starts = np.sort(rng.integers(max_start_time, size=2))
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
    for i in range(len(clip_filenames)):
        save_audio_clip(url, sample_rate, clip_filenames[i], start=starts[i],
                        interval=interval)
    return clip_filenames
