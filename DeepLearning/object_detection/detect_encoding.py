from __future__ import division

from models import *
from yolo_utils import *
from datasets import *

import os
import sys
import time
import datetime
import argparse
import math
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
sys.path.append(os.path.abspath(os.path.join('..')))
import utils

#num_tiles = utils.get_num_tiles()
num_tiles = 8

def get_tile(x,y):
    ratio = 1/num_tiles
    x_ = math.floor(x/ratio)
    y_ = math.floor(y/ratio)
    return [x_, y_]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="C:\\Users\\omossad\\Desktop\\dataset\\model_data\\raw_frames\\amin\\ma_2\\", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="base_model.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="tiny_yolo.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.05, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--out_folder", type=str, default="C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ma_2\\cave\\", help="path to output_folder")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    labels_file ='C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\pa_8.txt'
    labels = np.loadtxt(labels_file)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    print(classes)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        #if batch_i > 60:
        #    break
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    print("\nSaving images:")
    # Iterate through images and save plot of detections

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        #if img_i > 10:
        #    break
        print("(%d) Image: '%s'" % (img_i, path))
        img_name = path.split("\\")[-1].split(".")[0]
        frame_num = int(img_name.replace('frame_',''))
        img_name = opt.out_folder + "roi" + str(frame_num) + '.txt'
        f = open(img_name, "w")
        #print(img_name)
        # Create plot
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, (1080,1920))
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                c_x = (x1+x2)/2.0
                c_y = (y1+y2)/2.0
                box_w = x2 - x1
                box_h = y2 - y1
                line_to_write = str(int(cls_pred)) + " " + str(x1.item()/1920.0) + " " + str(y1.item()/1080.0) + " "
                line_to_write = line_to_write + str(box_w.item()/1920.0) + " " + str(box_h.item()/1080.0)
                f.write(line_to_write + '\n')


        f.close()
