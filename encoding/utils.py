import csv
import numpy as np
import configparser
import math
import os

config = configparser.ConfigParser()
config.read(['config.ini'])

def get_fps():
	fps_r = float(config.get("data", "fps_r"))
	fps_g = float(config.get("data", "fps_g"))
	return [fps_r, fps_g]

def get_path():
	return str(config.get("data", "data_path"))


def get_game_list():
	return ['fifa', 'csgo']

def listdirs(folder):
	return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def get_model_input_params():
	ipt_len = int(config.get("model", "ipt_len"))
	opt_len = int(config.get("model", "opt_len"))
	return [ipt_len, opt_len]

def get_frame_no(frame_name):
### FUNCTION TO RETRUN FRAME NUMBER FROM ITS NAME ###
    frame_without_ext = frame_name.split('.')[0]
    frame_no = frame_without_ext.split('_')[1]
    # Substracting 1 because frames start from 1
    return int(frame_no)-1
