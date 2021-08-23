import csv
import numpy as np
import configparser
import math
import os

config = configparser.ConfigParser()
config.read(['config.ini'])
######################################################################################
## [DATA] ##

def get_img_dim():
	W = float(config.get("data", "W"))
	H = float(config.get("data", "H"))
	return [W,H]

def get_fps():
	return float(config.get("data", "fps"))

def get_eye_blink_threshold():
	return int(config.get("data", "eye_blink_thre"))

def get_path():
	return str(config.get("data", "data_path"))

def get_game_list():
	return [g.strip() for g in config.get("data", "games").split(',')]
	#return ['fifa', 'csgo', 'nba', 'nhl']

def get_data(game_name):
	file_path = os.path.join(get_path(), 'public', game_name + '.txt')
	with open(file_path) as file:
		file_list = [l.strip() for l in file]
	return file_list

def listdirs(folder):
	return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

######################################################################################
## [MODEL] ##

def get_model_input_params():
	ipt_len = int(config.get("model", "ipt_len"))
	opt_len = int(config.get("model", "opt_len"))
	return [ipt_len, opt_len]


def get_encoder_params():
	delta_h = int(config.get("encoder", "delta_h"))
	delta_w = int(config.get("encoder", "delta_w"))
	sigma   = int(config.get("encoder", "sigma"))
	return [delta_h, delta_w, sigma]

def get_frame_no(frame_name):
### FUNCTION TO RETRUN FRAME NUMBER FROM ITS NAME ###
    frame_without_ext = frame_name.split('.')[0]
    frame_no = frame_without_ext.split('_')[1]
    # Substracting 1 because frames start from 1
    return int(frame_no)-1
