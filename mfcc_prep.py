import os
import glob
import torchaudio
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
# import s3prl
# from s3prl.nn import S3PRLUpstream

# import joblib
import torch
# import pandas as pd

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000
NUM_SAMPLE = 16000 * 3

transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=13,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)



print('START !!!')

for whole_video_path in tqdm( glob.glob('./videos/*') ):

    name = whole_video_path.split('/')[-1][:-4] + '.wav'

    video = VideoFileClip(whole_video_path)
    audio = video.audio
    audio.write_audiofile(f'./audio_files_mfcc/{name}')

    ori_audio, ori_sample_rate = torchaudio.load(f'./audio_files_mfcc/{name}', normalize=True)
    transform = torchaudio.transforms.Resample(ori_sample_rate, SAMPLE_RATE)
    audio = transform(ori_audio)



    if os.path.exists(f'./train/seg/{name[:-4]}_seg.csv'):
        with open(f'./train/seg/{name[:-4]}_seg.csv') as dur_file:

            a = 1
            infos = dur_file.readlines()
            while a < len(infos):
                comb = infos[a]
                a += 1
                comb = comb[:-1]

                person_id, start_id, end_id, tf = comb.split(',')

                start_frame = int(start_id)
                end_frame = int(end_id)

                # =====
                range = end_frame - start_frame
                if range > 30 * 4:
                    start_frame = start_frame + int(range/2) - 30 * 2
                    end_frame = start_frame + 30 * 4
                elif range < 30 * 4:
                    if start_frame + int(range/2) - 30 * 2 > 0:
                        start_frame = start_frame + int(range/2) - 30 * 2
                        end_frame = start_frame + 30 * 4
                    elif end_frame - int(range/2) + 30 * 2 > 8999:
                        start_frame = 8999 - 30 * 4
                        end_frame = 8999
                    else:
                        start_frame = 1
                        end_frame = 1 + 30 * 4
                # =====

                onset = int(start_frame / 30 * SAMPLE_RATE)
                offset = int(end_frame / 30 * SAMPLE_RATE)
                
                crop_audio = audio[:, onset:offset]

                # === MFCC ===
                mfcc = transform(crop_audio)
                np.save(f'./mfcc/{name[:-4]}_{start_id}_{end_id}', mfcc)

    elif os.path.exists(f'./test/seg/{name[:-4]}_seg.csv'):
        with open(f'./test/seg/{name[:-4]}_seg.csv') as dur_file:

            b = 1
            infos = dur_file.readlines()
            while b < len(infos):
                comb = infos[b]
                b += 1
                comb = comb[:-1]
                
                person_id, start_id, end_id = comb.split(',')

                start_frame = int(start_id)
                end_frame = int(end_id)

                # =====
                range = end_frame - start_frame
                if range > 30 * 4:
                    start_frame = start_frame + int(range/2) - 30 * 2
                    end_frame = start_frame + 30 * 4
                elif range < 30 * 4:
                    if start_frame + int(range/2) - 30 * 2 > 0:
                        start_frame = start_frame + int(range/2) - 30 * 2
                        end_frame = start_frame + 30 * 4
                    elif end_frame - int(range/2) + 30 * 2 > 8999:
                        start_frame = 8999 - 30 * 4
                        end_frame = 8999
                    else:
                        start_frame = 1
                        end_frame = 1 + 30 * 4
                # =====

                onset = int(start_frame / 30 * SAMPLE_RATE)
                offset = int(end_frame / 30 * SAMPLE_RATE)
                
                crop_audio = audio[:, onset:offset]

                # === MFCC ===
                mfcc = transform(crop_audio)
                np.save(f'./mfcc/{name[:-4]}_{start_id}_{end_id}', mfcc)