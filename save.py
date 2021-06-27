from datetime import datetime

import torch

def save(model, fpath):
	now = datetime.now()
	timestamp = now.strftime("%d%m-%H%M")

	save_folder = './models'
	save_path = f'{save_folder}/{fpath}-{timestamp}'

	torch.save(model.state_dict(), save_path) # model is saved in save_folder for reproducibility
	print(f'Model saved in {save_path}.')
	return save_path
