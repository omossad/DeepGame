import utils
import os
import csv
import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
import cv2

print('This file is for test purposes only')
def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        print('GT MAP VALUES')
        print(original_map.shape)
        print(np.max(original_map))
        print(np.sum(np.where(original_map > 0, 1, 0))/(480*640)*100)


        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32)
        ims[i, 0] /= 255.0
    return ims

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))
    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image.astype('float')
#     cv2 : BGR
#     PIL : RGB
    ims = ims[...,::-1]
    ims /= 255.0
    ims = np.rollaxis(ims, 3, 1)
    return ims

shape_r = 240
shape_c = 320
shape_r_gt = 30
shape_c_gt = 40
base_dir = 'C:\\Users\\omossad\\Desktop\\salicon\\'
imgs_val_path = base_dir + 'images\\val\\'
maps_val_path = base_dir + 'maps\\val\\'


counter = 0
def generator(b_s):
    images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
    maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.png')]
    images.sort()
    maps.sort()
    images , maps = shuffle(images,maps)
    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        if counter + b_s >= len(images):
          break
        counter = counter + b_s

counter = 0

for i,gt_map in generator(1):
    if counter > 1:
        break
    print ("Original")
    print(i[0].shape)
    org_image = i[0].copy()
    print(org_image.shape)
    org_image = np.rollaxis(org_image, 0, 3)
    print(org_image.shape)
    print("GT")
    print(gt_map.shape)
    print(np.max(gt_map[0][0]))
    #print(gt_map[0][0].tolist())
    print(gt_map[0][0].shape)
    counter = counter + 1

W = 1920
H = 1080
fix_arr = np.random.randn(5,3)
fix_arr -= fix_arr.min()
fix_arr /= fix_arr.max()
fix_arr[:,0] *= W
fix_arr[:,1] *= H

print(fix_arr)
def GaussianMask(sizex,sizey, sigma=33, center=None):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
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



[delta_h, delta_w, sigma] = utils.get_encoder_params()


sample_map = np.zeros((delta_h, delta_w))
fixation_x = 10
fixation_y = 10
sample_map += GaussianMask(delta_w, delta_h, sigma, (fixation_x,fixation_y))


# Normalization
sample_map = sample_map/np.amax(sample_map)
sample_map = sample_map*255
sample_map = sample_map.astype("uint8")

print(sample_map.shape)
print(sample_map)
cv2.imshow('image',sample_map)
cv2.waitKey(0)
resized = cv2.resize(sample_map, (1920,1080), interpolation = cv2.INTER_AREA)
cv2.imshow('image',resized)
cv2.waitKey(0)

#cv2.imshow('GT MAP', gt_map[0][0])
#cv2.waitKey(0)
print(sample_map.shape)
print(gt_map[0][0].shape)
print(np.sum(np.where(sample_map > 0, 1, 0))/(delta_h*delta_w)*100)
print(np.sum(np.where(gt_map[0][0] > 0, 1, 0))/(30*40)*100)
print(np.max(gt_map[0][0]))
print(np.max(sample_map))

'''
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

arr = [0.03223,0.62598,0.95215,1.16602,1.46582,1.83984,2.25049]
print(find_nearest(arr,0.5))

data_path = utils.get_path()
games = utils.get_game_list()

#games = [games[0]]

print('Read Frame Fixations')
players_list = utils.listdirs(data_path + 'raw_data\\')
for game in games:
    print('---------------------------------------')
    print('Processing ' + game + ' fixations')
    print('---------------------------------------')
    file_list = utils.get_data(game)

    #file_list  = [file_list[0]]


    for file in file_list:
        print('Current file is: ' + file)
        for x in players_list:
            if x.startswith(file[0:2]):
                player_name = x
        fixation_file_name = 'User '+ file[3:]
        input_file = data_path + 'raw_data\\' + player_name + '\\result\\' + fixation_file_name + '_fixations.csv'
        print(input_file)
        with open(input_file) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            data = [r[5:11] for r in reader]
            for i in data:
                if float(i[0]) < 0 or float(i[1]) < 0:
                    print(i[0])
                    print(i[1])
                    print('negative')
                if float(i[0]) > 1 or float(i[1]) > 1:
                    print('out of range')
                    print(i[0])
                    print(i[1])
            print(data)
            #print(data[:,0])
            dat = np.asarray(data)[:,0]
            #print(dat)
'''
'''
#### COUNT FRAMES IN Video
# import the necessary packages
import cv2
# grab a pointer to the video file and initialize the total
# number of frames read
path = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_data\\amgad\\user\\0000-scrn.avi'
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break

cap.release()

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print('Frame %d difference:'%i, abs(ts - cts))
print(calc_timestamps)
print(timestamps)
'''
