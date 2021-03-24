import pandas as pd
from audio_processor import AudioProcessor


def run_processor():
    df_episodes = pd.read_csv('data/episodes.csv')
    audio_processor = AudioProcessor()
    audio_processor.process(df_episodes[['audio', 'audio_length', 'uuid']][3:7])


if __name__ == '__main__':
    run_processor()
