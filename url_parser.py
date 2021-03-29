import numpy as np
import subprocess
from tqdm import tqdm

ROOT_DIR = '/content/drive/MyDrive/Podcast_Audio_Classification'   


def drop_invalid_urls(df, start):
    # invalidate examples with broken links
    for index, url in tqdm(df[start:].audio.items(), desc='UrlParser'):
        url_checker = subprocess.run(['ffprobe', '-loglevel', 'quiet', url])
        if url_checker.returncode != 0:
            df.at[index, 'audio'] = np.nan
        else:
            break

    # drop invalidated examples
    return df.dropna(subset=['audio']).reset_index(drop=True), index
