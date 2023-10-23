import os
import glob
import torchaudio
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip

# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# signal(SIGPIPE, SIG_IGN)

print('START !!!')

# for whole_video_path in tqdm( glob.glob('./videos/*') ):

#     name = whole_video_path.split('/')[-1][:-4] + '.wav'
#     # print(name)

#     if os.path.exists(f'./audio_files/{name}') : continue

#     # with open('mfcc_output', 'a') as a:
#     #     a.write( name )
#     #     a.write('\n')

#     try:
#         video = VideoFileClip(whole_video_path)
#         audio = video.audio
#         audio.write_audiofile(f'./audio_files/{name}')
#     except:
#         print(f'can not write {name}')

# print('finish store !')

for whole_video_path in tqdm( glob.glob('./videos/*') ):

    name = whole_video_path.split('/')[-1][:-4] + '.wav'

    try:
        ori_audio, ori_sample_rate = torchaudio.load(f'./audio_files/{name}', normalize=True)
    except:
        print(f'pass {name} !')
        continue

    # os.remove(name)

    sample_rate = 16000
    transform = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)
    audio = transform(ori_audio)
    # print(audio.shape)

    if os.path.exists(f'./train/seg/{name[:-4]}_seg.csv'):
        with open(f'./train/seg/{name[:-4]}_seg.csv') as dur_file:
            infos = dur_file.readlines()
            for info in range(len(infos)-1):
                person_id, start_id, end_id, tf = infos[info+1].split(',')

                start_frame = int(start_id)
                end_frame = int(end_id)
                onset = int(start_frame / 30 * sample_rate)
                offset = int(end_frame / 30 * sample_rate)
                # print(offset-onset)
                if offset-onset == 0 : continue
                crop_audio = audio[:, onset:offset]
                # print(crop_audio.shape)
                if crop_audio.shape[1] == 0 : continue

                transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=13,
                    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
                )
                mfcc = transform(crop_audio)
                # print(mfcc.shape)

                try:
                    np.save(f'./mfcc_frame/{name[:-4]}_{start_id}_{end_id}', mfcc)
                except:
                    print(f'can not save {name[:-4]}_{start_id}_{end_id}')

    elif os.path.exists(f'./test/seg/{name[:-4]}_seg.csv'):
        with open(f'./test/seg/{name[:-4]}_seg.csv') as dur_file:
            infos = dur_file.readlines()
            for info in range(len(infos)-1):
                person_id, start_id, end_id = infos[info+1].split(',')

                start_frame = int(start_id)
                end_frame = int(end_id)
                onset = int(start_frame / 30 * sample_rate)
                offset = int(end_frame / 30 * sample_rate)
                # print(offset-onset)
                if offset-onset == 0 : continue
                crop_audio = audio[:, onset:offset]
                # print(crop_audio.shape)
                if crop_audio.shape[1] == 0 : continue

                transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=13,
                    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
                )
                mfcc = transform(crop_audio)
                # print(mfcc.shape)

                try:
                    np.save(f'./mfcc_frame/{name[:-4]}_{start_id}_{end_id}', mfcc)
                except:
                    print(f'can not save {name[:-4]}_{start_id}_{end_id}')

    else: raise('wrong file name !')