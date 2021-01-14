import utils
import os
import csv
import pandas as pd


games = utils.get_game_list()
data_path = utils.get_path()
#ffmpeg.exe -i C:\Users\omossad\Desktop\dataset\raw_frames\nba\am_0\0000-scrn.avi -vf fps=10 C:\Users\omossad\Desktop\dataset\raw_frames\nba\am_0\frame_%05d.jpg

class FrameInfo:
    def __init__(self, frameNum, xFixation, yFixation, sceneType):
        self.frameNum  = frameNum
        self.xFixation = xFixation
        self.yFixation = yFixation
        self.sceneType = sceneType
        label_path = utils.get_path() + 'raw_data\\'


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
            video_name = format(int(file[3:]), '04d')
            input_video = data_path + 'raw_data\\' + player_name + '\\user\\' + video_name + '-scrn.avi'
            output_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            os.makedirs(output_path, exist_ok=True)
            cmd = 'ffmpeg.exe -i '
            cmd = cmd + input_video
            cmd = cmd + ' -vf fps=10 '
            cmd = cmd + output_path
            cmd = cmd + 'frame_%05d.jpg'
            print(cmd)
            os.system(cmd)

def process_fixation(data, num_frames):
    fps = utils.get_fps()
    frame_no  = 0
    iter = 0
    fixations = []
    while frame_no < num_frames:
        frame_time = frame_no * 1/fps
        if frame_time < data[iter][2] + data[iter][3]:
            fixations.append([frame_no, data[iter][0], data[iter][1], data[iter][-1]])
            frame_no = frame_no + 1
        else:
            iter = iter + 1
    print(num_frames)
    print(fixations)
    #nme = ["aparna", "pankaj", "sudhir", "Geeku"]
    #deg = ["MBA", "BCA", "M.Tech", "MBA"]
    #scr = [90, 40, 80, 98]

    # dictionary of lists
    #dict = {'name': nme, 'degree': deg, 'score': scr}
    #for i in range(num_frames):
    #    print('sico')

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
            input_file = data_path + 'raw_data\\' + player_name + '\\result\\' + fixation_file_name + '_fixations.csv'
            output_file = data_path + 'raw_labels\\' + game + '\\' + file + '.txt'
            frames_path = data_path + 'raw_frames\\' + game + '\\' + file + '\\'
            num_frames = len(os.listdir(frames_path))
            with open(input_file) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                data = [float(r[5:11]) for r in reader]
                if file == 'am_0':
                    process_fixation(data, num_frames)



            #df = pd.DataFrame(dict)

            # saving the dataframe
            #df.to_csv('file1.csv')

if __name__ == "__main__":
    #create_frames()
    create_labels()
