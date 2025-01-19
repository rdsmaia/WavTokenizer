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
	parser.add_argument('--codes_path', type=str, required=True,
				help='location of code files.')
	parser.add_argument('--output_path', type=str, required=True,
				help='location to save synthesized audio.')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}\n')
	if device.type == 'cuda':
        	print(f'Using {torch.cuda.get_device_name(0)}')

	# load model
	config_path='configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
	model_path='/home/rmaia/.cache/huggingface/hub/models--novateur--WavTokenizer-large-unify-40token/snapshots/7dc47a9450ce71cab736434b7645604013ca028b/wavtokenizer_large_unify_600_24k.ckpt'
	wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
	wavtokenizer = wavtokenizer.to(device)

	# output dir
	os.makedirs(args.output_path, exist_ok=True)

	# filelist
	filelist = glob(args.codes_path+'/*.pth')
	print(f'Number of files to be processed: {len(filelist)}')

	# process files
	for file in tqdm(filelist, total=len(filelist)):

		# load codes
		audio_tokens = torch.load(file, map_location=device).unsqueeze(0)

		# audio_tokens [n_q,1,t]/[n_q,t] into features
		features = wavtokenizer.codes_to_features(audio_tokens)
		bandwidth_id = torch.tensor([0]).to(device)

		# synthesizes speech
		audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

		# save audio
		filename = os.path.join(args.output_path, os.path.basename(file).replace('.pth', '.wav'))
		torchaudio.save(filename, audio_out.detach().cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=24)


