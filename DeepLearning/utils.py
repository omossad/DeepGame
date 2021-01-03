import csv
import numpy as np
import configparser
import math
import os

config = configparser.ConfigParser()
config.read(['config.ini'])

def get_img_dim():
	W = float(config.get("data", "W"))
	H = float(config.get("data", "H"))
	return [W,H]

def get_fps():
	return float(config.get("data", "fps"))

def get_path():
	return str(config.get("data", "data_path"))

def get_data(game_name):
	file_path = get_path() + 'raw_data\\' + game_name + '.txt'
	with open(file_path) as file:
		file_list = [i.strip() for i in file]
	return file_list

def get_game_list():
	return ['fifa', 'csgo', 'nba', 'nhl']

def listdirs(folder):
	return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
