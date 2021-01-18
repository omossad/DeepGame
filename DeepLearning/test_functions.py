import utils
import os
import csv
import  as pd
import numpy as np
import pandas as pd

data_path = utils.get_path()
games = utils.get_game_list()
print('Read Frame Fixations')
players_list = utils.listdirs(data_path + 'raw_data\\')
for game in games:
    print('---------------------------------------')
    print('Processing ' + game + ' fixations')
    print('---------------------------------------')
    file_list = utils.get_data(game)
    for file in file_list:
        print('Current file is: ' + file)
        for x in players_list:
            if x.startswith(file[0:2]):
                player_name = x
        output_file = data_path + 'raw_labels\\' + game + '\\' + file + '.csv'
        test = pd.read_csv('data.csv')
        print(test)
