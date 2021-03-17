import utils
import os
import csv
import pandas as pd
import numpy as np
import math

games = utils.get_game_list()
'''
LIMIT ONE GAME FOR NOW
games = [games[0]]

'''



data_path = utils.get_path()

#ffmpeg.exe -i C:\Users\omossad\Desktop\dataset\raw_frames\nba\am_0\0000-scrn.avi -vf fps=10 C:\Users\omossad\Desktop\dataset\raw_frames\nba\am_0\frame_%05d.jpg

#class FrameInfo:
#    def __init__(self, frameNum, xFixation, yFixation, sceneType):
#        self.frameNum  = frameNum
#        self.xFixation = xFixation
#        self.yFixation = yFixation
#        self.sceneType = sceneType
#        label_path = utils.get_path() + 'raw_data\\'


def create_frames():
    print('Split Video Sequences into Frames \n')
    players_list = utils.listdirs(data_path + 'raw_data\\')
    for game in games:
        print('---------------------------------------')
        print('Processing ' + game + ' video sequences')
        print('---------------------------------------')
        file_list = utils.get_data(game)
        for file in file_list:
            print('Current file is: ' + file)
            for x in players_list:
                if x.startswith(file[0:2]):
                    player_name = x

            #video_name = format(int(file[3:]), '04d')
            #input_video = data_path + 'raw_data\\' + player_name + '\\user\\' + video_name + '-scrn.avi'
            input_video = data_path + 'raw_data\\' + player_name + '\\result\\' + file + '.avi'
            output_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            os.makedirs(output_path, exist_ok=True)
            cmd = 'ffmpeg.exe -i '
            cmd = cmd + input_video
            cmd = cmd + ' -vf fps=10 '
            cmd = cmd + output_path
            cmd = cmd + 'frame_%05d.jpg'
            print(cmd)
            os.system(cmd)


def find_nearest(array,value):
### FUNCTION TO FIND THE INDEX OF THE NEAREST VALUE FROM AN ARRAY ###
# source: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array #
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def process_fixation(f_data, g_data, num_frames):
    gaze_times = np.asarray(g_data)[:,0]
    fixation_ids = np.asarray(f_data)[:,2]
    fps = utils.get_fps()
    frame_no  = 0
    fixations = []
    while frame_no < num_frames:
        #adjust_offset = 1439/1414.0 # this value to offset difference between Gazepoint software and raw video
        frame_time = frame_no * 1/fps
        #frame_time = frame_no * 1/fps * adjust_offset
        #print(frame_time)
        # find the nearest gaze reading to the frame time
        g_idx = find_nearest(gaze_times, frame_time)

        # find the fixation relative to the gaze reading (-1 because fixations start from 1)
        f_idx = g_data[g_idx][1]
        if f_idx > len(f_data):
            # Handle when the fixation ID is out of range
            # Set as last fixation
            f_idx = len(f_data) - 1
        # Create fixation entry (X , Y, Valid)
        idx = np.where(fixation_ids == f_idx)
        if len(idx[0]) < 1:
            idx = f_idx
        else:
            idx = idx[0][0]
        fixations.append([f_data[idx][0], f_data[idx][1], g_data[g_idx][2]])
        frame_no = frame_no + 1

        '''
        ### OLD FIXATIONS CODE (IGNORE)
        if frame_time < float(data[iter][2]) + float(data[iter][3]):
            fixations.append([float(data[iter][0]), float(data[iter][1]), int(data[iter][-1])])
            frame_no = frame_no + 1
        else:
            if iter < len(data) - 1:
                iter = iter + 1
            else:
                data[iter][3] = 99
        '''
    fixs = np.array(fixations)
    return {'pos_x': fixs[:,0], 'pos_y': fixs[:,1], 'valid': fixs[:,2].astype(int)}


def create_labels():
    print('Find Frame Fixations')
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
            fixation_file_name = 'User '+ file[3:]
            gaze_file = data_path + 'raw_data\\' + player_name + '\\result\\' + fixation_file_name + '_all_gaze.csv'
            fixation_file = data_path + 'raw_data\\' + player_name + '\\result\\' + fixation_file_name + '_fixations.csv'
            output_file = data_path + 'raw_labels\\' + game + '\\' + file + '.csv'
            frames_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            num_frames = len(os.listdir(frames_path))
            with open(fixation_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                # DATA IS:    FPOGX,      FPOGY,        FPOGID #
                f_data = [ [float(r[5]), float(r[6]) , int(r[9])] for r in reader]
            with open(gaze_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                # DATA IS:   TIME,        FPOGID,       FPOGV #
                g_data = [[float(r[3]) , int(r[9]) , int(r[10])] for r in reader]
            fixations = process_fixation(f_data, g_data, num_frames)
            # SAVED FORMAT: FPOGID, FPOGX, FPOGY, FPOGV
            pd.DataFrame(fixations).to_csv(output_file)

def refine_labels():
    print('Not supported yet')

def create_overlap(arr, n_items, n_overlap):
### FUNCTION TO SPLIT LIST INTO OVERLAPPING SUB-LISTS ###
# source:     https://stackoverflow.com/questions/36586897/splitting-a-python-list-into-a-list-of-overlapping-chunks #
    out_arr = [arr[i:i+n_items] for i in range(0, len(arr), n_overlap)]
    to_remove = []
    for i in out_arr:
        if len(i) < n_items:
            to_remove.append(i)
    for i in to_remove:
        out_arr.remove(i)
    return out_arr


def create_model_inputs():
    print('CREATING MODEL INPUTS')
    [ipt_len, opt_len] = utils.get_model_input_params()
    players_list = utils.listdirs(data_path + 'raw_data\\')
    for game in games:
        print('---------------------------------------')
        print('Processing ' + game + ' model inputs')
        print('---------------------------------------')
        file_list = utils.get_data(game)

        for file in file_list:
            print('Current file is: ' + file)
            labels_file = data_path + 'raw_labels\\' + game + '\\' + file + '.csv'
            frames_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            output_file = data_path + 'model_inputs\\' + game + '\\' + file + '.csv'
            frame_names = os.listdir(frames_path)
            num_frames = len(frame_names)
            model_inputs = create_overlap(frame_names, ipt_len + opt_len, opt_len)
            #print(model_inputs)
            with open(labels_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                # DATA IS:    FPOGX,      FPOGY,        FPOGID #
                l_data = [r for r in reader]

            for i in model_inputs:
                to_substitute = []
                for j in range(opt_len):
                    to_substitute.append(i[len(i)-opt_len+j])
                for j in to_substitute:
                    i.remove(j)
                    i.append(l_data[utils.get_frame_no(j)][1:3])
            ## DATA IS: Frame Names [1: input_len] , Fixations (X,Y) [1: output_len]

            pd.DataFrame(model_inputs).to_csv(output_file)


if __name__ == "__main__":
    #create_frames()
    create_labels()
    #refine_labels()
    #create_model_inputs()
