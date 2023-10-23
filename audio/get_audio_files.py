import os
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip

print('start generate audio files')

video_root = os.path.abspath('./student_data/videos/')
audio_root = os.path.abspath('./student_data/audio_files/')

if not os.path.exists(audio_root):
    os.makedirs(audio_root, exist_ok=True)

for whole_video_path in tqdm(glob.glob(os.path.join(video_root, "*"))):
    path = os.path.abspath(whole_video_path)
    name = whole_video_path.split('/')[-1][:-4] + '.wav'

    if os.path.exists(os.path.join(audio_root, name)) : continue

    try:
        video = VideoFileClip(path)
        audio = video.audio
        audio.write_audiofile(os.path.join(audio_root, name))
    except:
        print(f'can not write {name}')

print('finished')