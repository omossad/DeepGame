import utils
import os
import csv
import pandas as pd
import numpy as np
import math
from itertools import groupby
from operator import itemgetter

games = utils.get_game_list()
#games = [games[0]]
data_path = utils.get_path()


def consecutive_occurences(arr):
### FUNCTION TO FIND THE CONSECUTIVE OCCURENCES OF ELEMENTS IN LIST ###
# source: https://stackoverflow.com/questions/52422866/how-to-find-the-maximum-consecutive-occurrences-of-a-number-in-python #
    print((values, key) for key, values in groupby(arr))
    num_times, occurrence = max((len(list(values)), key) for key, values in groupby(arr))
    print("%d occurred %d times" % (occurrence, num_times))
    #return [num_times, occurence]


def consecutive_integers(my_list):
### FUNCTION TO FIND THE CONSECUTIVE INTEGERS IN LIST ###
# source: https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list #
    """will split the list base on the index"""
    my_index = [(x+1) for x,y in zip(my_list, my_list[1:]) if y-x != 1]
    output = list()
    prev = 0
    for index in my_index:
        new_list = [ x for x in my_list[prev:] if x < index]
        output.append(new_list)
        prev += len(new_list)
    output.append([ x for x in my_list[prev:]])
    return output
'''
def consecutive_integers(nums):
### FUNCTION TO FIND THE CONSECUTIVE INTEGERS IN LIST ###
# source: https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list #
    ranges = sum((list(t) for t in zip(nums, nums[1:]) if t[0]+1 != t[1]), [])
    iranges = iter(nums[0:1] + ranges + nums[-1:])
    print(', '.join([str(n) + '-' + str(next(iranges)) for n in iranges]))
    #for k, g in groupby(enumerate(data), lambda(i, x): i-x):
    #    print map(itemgetter(1), g)
'''

def analyze():
    print('Find Invalid Fixations')
    players_list = utils.listdirs(data_path + 'raw_data\\')
    for game in games:
        print('---------------------------------------')
        print('Processing ' + game )
        print('---------------------------------------')
        file_list = utils.get_data(game)
        #file_list = [file_list[0]]
        for file in file_list:
            print('Current file is: ' + file)
            fixation_file = data_path + 'raw_labels\\' + game + '\\' + file + '.csv'
            frames_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            num_frames = len(os.listdir(frames_path))
            with open(fixation_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                # DATA IS:    FPOGX,      FPOGY,        FPOGID #
                data = [ r for r in reader]
                valid_flags = [int(d[3]) for d in data]
                invalid_idx = []
                for i in range(len(valid_flags)):
                    if valid_flags[i] == 0:
                        invalid_idx.append(i)
                consctv_lst = consecutive_integers(invalid_idx)
                consctv_lst_len = []    # VARIABLE USED FOR ANALYSIS ONLY
                for i in consctv_lst:
                    if len(i) > 1:
                        consctv_lst_len.append(len(i))
                    if len(i) > 10:
                        #print(i)
                        #print(data[i[0]-1])
                        #print(data[i[0]])
                        #if len(data) > i[-1]+1:
                        #    print(data[i[-1]+1])
                        print(file)
                        # FILES TO EXCLUDE: pa_4, pu_0, pu_5, pu_6, pu_7
                print(consctv_lst_len)
            #print(valid_flags)
            #print(len(data))
            #print(data)

if __name__ == "__main__":
    #create_frames()
    analyze()
