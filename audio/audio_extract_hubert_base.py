import os
import glob
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from s3prl.nn import S3PRLUpstream

parser = argparse.ArgumentParser()
parser.add_argument(
	'--final_data_dir',
	type=str,
	default='./student_data/',
	help='Dataset Root. (default: ./student_data/)'
)
parser.add_argument(
	'--final_audio_dir',
	type=str,
	default='./student_data/audio_files/',
	help='Audio Files Root. (default: ./student_data/audio_files/)'
)
parser.add_argument(
	'--audio_sample_seconds',
	type=int,
	default=5,
	help='Extracted Audio Length. (default: 5 seconds)'
)
args = parser.parse_args()

data_root = os.path.abspath(args.final_data_dir)	### dataset root
audio_root = os.path.abspath(args.final_audio_dir)	### audio files root

SAMPLE_RATE = 16000
NUM_SAMPLE = SAMPLE_RATE * args.audio_sample_seconds

def pad_truncate(audio):
	"""pad/truncate if < or > NUM_SAMPLE"""

	# combine source
	audio = torch.mean(audio, 0).unsqueeze(0)	# size: (1, seq_len)

	if audio.shape[1] > NUM_SAMPLE:
		audio = audio[:, :NUM_SAMPLE]
	elif audio.shape[1] < NUM_SAMPLE:
		ori_len = audio.shape[1]
		padded_sample_len = NUM_SAMPLE - ori_len
		last_dim_padding = (0, padded_sample_len)	# (left prepend, right padding)
		audio = torch.nn.functional.pad(audio, last_dim_padding)

	return audio

if __name__ == "__main__":
	if not os.path.exists('./hubert_features/'):
		os.makedirs('./hubert_features/', exist_ok=True)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('Device used:', device)

	model = S3PRLUpstream("hubert", refresh=False).to(device)
	model.eval()
	print(model)

	print('===	START preprocess	===')

	for whole_video_path in tqdm(glob.glob(os.path.join(data_root, 'videos', "*"))):

		name = whole_video_path.split('/')[-1][:-4] + '.wav'

		try:
			ori_audio, ori_sample_rate = torchaudio.load(os.path.join(audio_root, name), normalize=True)
		except:
			print(f'pass {name} !')
			continue

		transform = torchaudio.transforms.Resample(ori_sample_rate, SAMPLE_RATE)
		audio = transform(ori_audio).to(device)

		if os.path.exists(os.path.join(data_root, "train", "seg", f'{name[:-4]}_seg.csv')):
			with open(os.path.join(data_root, "train", "seg", f'{name[:-4]}_seg.csv')) as dur_file:
				infos = dur_file.readlines()
				for info in range(len(infos)-1):
					person_id, start_id, end_id, tf = infos[info+1].split(',')

					start_frame = int(start_id)
					end_frame = int(end_id)
					onset = int(start_frame / 30 * SAMPLE_RATE)
					offset = int(end_frame / 30 * SAMPLE_RATE)

					if offset-onset == 0 : 
						print('offset-onset == 0')
						continue

					crop_audio = audio[:, onset:offset]

					if crop_audio.shape[1] == 0 : 
						print('crop_audio.shape[1] == 0')
						continue

					# TODO: Data aug
					# transform = torchaudio.transforms.MFCC(
					#     sample_rate=SAMPLE_RATE,
					#     n_mfcc=13,
					#     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
					# )
					# mfcc = transform(crop_audio)
					# # print(mfcc.shape)

					# TODO: s3prl
					crop_audio = torch.Tensor(crop_audio).to(device)
					crop_audio = pad_truncate(crop_audio)

					print(crop_audio.shape)
					crop_audio_len = torch.LongTensor([NUM_SAMPLE]).to(device)
					print(crop_audio_len)

					with torch.no_grad():
						all_hs, all_hs_len = model(crop_audio, crop_audio_len)
						print("test")

						# last layer
						hs = all_hs[-1]
						hs_len = all_hs_len[-1]
						print(hs.shape, hs_len.shape)
						assert hs_len.dim() == 1

						try:
							torch.save(hs, f'./hubert_features/{name[:-4]}_{start_id}_{end_id}.feat')
						except:
							print(f'can not save {name[:-4]}_{start_id}_{end_id}.feat')