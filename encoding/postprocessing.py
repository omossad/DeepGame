import numpy as np

path='C:\\Users\\omossad\\Desktop\\codes\\ROI-PyTorch\\DeepGame\\visualize\\'
pre_arr = np.loadtxt(path+'nhl_predicted.txt')
gt_arr = np.loadtxt(path+'nhl_labels.txt')

W = 1920.0
H = 1080.0
num_tiles = 8
tile_w = 1/num_tiles
tile_h = 1/num_tiles

def tile_to_coor(tile_x, tile_y):
    coor_x = tile_x*tile_w
    coor_y = tile_y*tile_h
    return [coor_x, coor_y, tile_w, tile_h]

def unmap_tile(arr):
    rois = []
    for i in range(len(arr)):
        if arr[i] > 0:
            tile_x = int(i/num_tiles)
            tile_y = int(i%num_tiles)
            rois.append(tile_to_coor(tile_x, tile_y))
    return rois

output_path ='D:\\Encoding\\nhl\\kh_9\\rois\\'
counter = 0
for i in range(len(pre_arr)):
    rois_gt = unmap_tile(gt_arr[i])
    rois_pr = unmap_tile(pre_arr[i])
    counter = counter + 2
    print(rois_gt)
print(counter)
