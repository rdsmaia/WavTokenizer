import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from glob import glob
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--audio_path', type=str, required=True,
				help='location of 24kHz audio files.')
	parser.add_argument('--output_path', type=str, required=True,
				help='location to save audio tokens.')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}\n')
	if device.type == 'cuda':
        	print(f'Using {torch.cuda.get_device_name(0)}')

	# load model
	config_path='configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
	model_path='/home/rdsmaia/.cache/huggingface/hub/models--novateur--WavTokenizer-large-unify-40token/snapshots/7dc47a9450ce71cab736434b7645604013ca028b/wavtokenizer_large_unify_600_24k.ckpt'
	wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
	wavtokenizer = wavtokenizer.to(device)

	# output dir
	os.makedirs(args.output_path, exist_ok=True)

	# filelist
	filelist = []
	for type in ['wav', 'flac', 'mp3']:
		filelist += glob(args.audio_path+'/**/*.'+type, recursive=True)
	print(f'Number of files to be processed: {len(filelist)}')

	# process files
	for file in tqdm(filelist, total=len(filelist)):

		wav, sr = torchaudio.load(file)
		wav = convert_audio(wav, sr, 24000, 1)
		bandwidth_id = torch.tensor([0])
		wav=wav.to(device)
		_, audio_tokens= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

		ext = file.split('.')[-1]
		filename = os.path.join(args.output_path, os.path.basename(file).replace('.'+ext,'.pth'))
		torch.save(audio_tokens.cpu().int().squeeze(), filename)


