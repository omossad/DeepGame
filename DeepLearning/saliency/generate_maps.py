### GENERATE MAPS ###
# Code to generate the fixation maps based on the lables
# Iterate over all frames and save the maps
import utils
import os
import csv
import pandas as pd
import numpy as np
import cv2

games = utils.get_game_list()
'''
LIMIT ONE GAME FOR NOW
'''
games = [games[0]]

data_path = utils.get_path()
[delta_h, delta_w, sigma] = utils.get_encoder_params()

def calculate_map(fixation_p):
    gt_map = np.zeros((delta_h, delta_w))
    gt_map += GaussianMask(delta_w, delta_h, sigma, (fixation_p))

    # Normalization
    gt_map = gt_map/np.amax(gt_map)
    gt_map = gt_map*255
    gt_map = gt_map.astype("uint8")
    return gt_map


def GaussianMask(sizex, sizey, sigma=8, center=None):
# GENERATE A GAUSSIAN MASK
# source: https://github.com/takyamamoto/Fixation-Densitymap/blob/master/Fixpos2Densemap.py
    """
    sizex  : image width
    sizey  : image height
    sigma  : gaussian Sd (8 for Visual Angle of 1 deg)
    center : gaussian mean (fixation point)
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey,sizex))

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)



def generate_maps():
    print('Generating Fixation Maps \n')
    players_list = utils.listdirs(data_path + 'raw_data\\')
    for game in games:
        print('---------------------------------------')
        print('Processing ' + game + ' video sequences')
        print('---------------------------------------')
        file_list = utils.get_data(game)
        #file_list = [file_list[0]]
        for file in file_list:
            print('Current file is: ' + file)
            labels_file = data_path + 'raw_labels\\' + game + '\\' + file + '.csv'
            frames_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            output_path = data_path + 'gt_maps\\' + game + '\\' + file + '\\'
            frame_names = os.listdir(frames_path)
            #print(frame_names)
            num_frames = len(frame_names)

            with open(labels_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                # DATA IS:    FPOGX,      FPOGY,        FPOGID #
                l_data = [r for r in reader]
            for f in frame_names:
                __ , fixation_x, fixation_y, __ = l_data[utils.get_frame_no(f)]
                fixation_x = int(float(fixation_x) * delta_w)
                fixation_y = int(float(fixation_y) * delta_h)
                gt_map = calculate_map([fixation_x, fixation_y])
                cv2.imwrite(output_path + f, gt_map)


if __name__ == "__main__":
    generate_maps()
