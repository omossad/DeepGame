### THIS CODE CALCULATES THE SALIENCY MAP ###
# source: # https://github.com/immortal3/MLNet-Pytorch/blob/master/MLNet_Pytorch.ipynb #
# dataset source: http://salicon.net/challenge-2017/ #


#!wget http://lsun.cs.princeton.edu/challenge/2015/eyetracking_salicon/data/image.zip
# unzipping data
#!unzip image.zip > unzipping.log
#!ls images | head -5

# installing gdown
# Reference : https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
#! pip install gdown

# Dowloading Ground Truth
#! gdown https://drive.google.com/uc?id=0B2hsWbciDVedNWJZMlRxeW1PY1U&export=download

# unzipping maps (ground truth)
#! unzip maps.zip > unzipping_maps.log

# unzipping creates two folders containing saliency maps
#  1. train
#  2. test
#!ls train | head -5



# moving testing data to another dir
import os
#import shutil

#images_files  = os.listdir('images')
#print ("Total Number of Images: {}".format(len(images_files)))

#!mkdir test_images
#!mkdir val_images

#cnt = 0

#for f in images_files:
#  if 'test' in f:
#    shutil.move('images/'+f,'test_images/')
#  elif 'val' in f:
#    shutil.move('images/'+f,'val_images/')


#! ls test_images | head -5
#! ls val_images | head -5

base_dir = 'C:\\Users\\omossad\\Desktop\\dataset\\'
print ("Total Train Images: {}".format(len(os.listdir(base_dir + 'raw_frames\\fifa\\am_0\\'))))
print ("Total Valid Images: {}".format(len(os.listdir(base_dir + 'raw_frames\\fifa\\am_1\\'))))
print ("Total Test  Images: {}".format(len(os.listdir(base_dir + 'raw_frames\\fifa\\am_2\\'))))



import cv2
import numpy as np

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


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32)
        ims[i, 0] /= 255.0
    return ims




#import os
from sklearn.utils import shuffle

imgs_train_path = base_dir + 'raw_frames\\fifa\\am_0\\'
maps_train_path = base_dir + 'gt_maps\\fifa\\am_0\\'


imgs_val_path = base_dir + 'raw_frames\\fifa\\am_1\\'
maps_val_path = base_dir + 'gt_maps\\fifa\\am_1\\'




def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.jpg')]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.jpg')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.jpg')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()

    images , maps = shuffle(images,maps)

    counter = 0

    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        if counter + b_s >= len(images):
          break
        counter = counter + b_s



import torch
import torch.nn as nn
import torchvision.models as models


class MLNet(nn.Module):

    def __init__(self,prior_size):
        super(MLNet, self).__init__()
        # loading pre-trained vgg16 model and
        # removing last max pooling layer
        features = list(models.vgg16(pretrained = True).features)[:-1]

        # making same spatial size
        # by calculation :)
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2

        self.features = nn.ModuleList(features).eval()
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64,1,kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1,1,prior_size[0],prior_size[1]), requires_grad=True))

        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)

    def forward(self, x):

        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)

        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0],results[1],results[2]),1)

        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)

        # 64 filters convolution layer
        x = self.int_conv(x)
        # 1*1 convolution layer
        x = self.pre_final_conv(x)

        upscaled_prior = self.bilinearup(self.prior)
        # print ("upscaled_prior shape: {}".format(upscaled_prior.shape))

        # dot product with prior
        x = x * upscaled_prior
        x = torch.nn.functional.relu(x,inplace=True)
        return x


# Modified MSE Loss Function
class ModMSELoss(torch.nn.Module):
    def __init__(self,shape_r_gt,shape_c_gt):
        super(ModMSELoss, self).__init__()
        self.shape_r_gt = shape_r_gt
        self.shape_c_gt = shape_c_gt

    def forward(self, output , label , prior):
        #print(output[0,0,:].shape)
        #print(torch.max(label[0,0,:]))
        prior_size = prior.shape
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt)
        #print(torch.max(output_max[0,0,:]))
        reg = ( 1.0/(prior_size[0]*prior_size[1]) ) * ( 1 - prior)**2
        loss = torch.mean( ((output / output_max) - label)**2 / (1 - label + 0.1) )  +  torch.sum(reg)
        return loss


# Input Images size
shape_r = 240
shape_c = 320
#shape_r = 480
#shape_c = 640


# Output Image size (generally divided by 8 from Input size)
shape_r_gt = 30
shape_c_gt = 40
#shape_r_gt = 60
#shape_c_gt = 80


last_freeze_layer = 23
# last_freeze_layer = 28

prior_size = ( int(shape_r_gt / 10) , int(shape_c_gt / 10) )

model = MLNet(prior_size).cuda()


# freezing Layer
for i,param in enumerate(model.parameters()):
  if i < last_freeze_layer:
    param.requires_grad = False


criterion = ModMSELoss(shape_r_gt,shape_c_gt).cuda()

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
import time
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

loss_history = []
nb_epochs = 10
batch_size = 16

for epoch in range(nb_epochs):
  t1 = time.time()
  image_trained = 0

  for i,gt_map in generator(batch_size):

      optimizer.zero_grad()
#       print (i.shape)

      i,gt_map = torch.tensor(i.copy(),dtype=torch.float),torch.tensor(gt_map,dtype=torch.float)
      for idx,x in enumerate(i):
        i[idx] = normalize(x)
      i,gt_map = i.cuda(),gt_map.cuda()

      image_trained += batch_size

      out = model.forward(i)
      loss = criterion(out,gt_map,model.prior.clone())
      loss.backward()
      optimizer.step()

      if image_trained % (batch_size*20) == 0:
        print ("Epcohs:{} Images:{} Loss:{}".format(epoch,image_trained,loss.item()) )
  t2 = time.time()
  time_per_epoch = (t2 - t1) / 60.0
  print ('Time taken for epoch-{} : {}m'.format(epoch,time_per_epoch))




import matplotlib.pyplot as plt

# how many pic you want to visualiz at randomly
no_visual = 2


visual_cnt = 0
for i,gt_map in generator(1,phase_gen='val'):
    print ("Original")
    org_image = i[0].copy()

    org_image = np.rollaxis(org_image, 0, 3)
    print (org_image.shape)
    i = torch.tensor(i.copy(),dtype=torch.float)
    for idx,x in enumerate(i):
        i[idx] = normalize(x)
    i = i.cuda()
    out = model.forward(i)
    # adding term which were subtracted at pre processing
    plt.imshow(org_image)
    plt.show()
    print ("predicted")
    plt.imshow(out[0].squeeze(0).data.cpu().numpy(),cmap='gray')
    plt.show()
    print ("Original")
    plt.imshow(gt_map[0][0],cmap='gray')
    plt.show()
    visual_cnt += 1
    if visual_cnt > no_visual:
      break

# creatin Dir for saving model
#!mkdir saved_models -p


import time
import datetime

# saving model weight
additional_info = '25_epochs_v3'
#full_path = base_dir +'model\\' + str(datetime.datetime.now()) + '_' + additional_info + '.model'
full_path = base_dir +'model\\' + additional_info + '.model'
torch.save(model.state_dict(), full_path)

# for loading Model
# model.load_state_dict(torch.load(PATH))

'''
test_file = [ f for f in os.listdir(base_dir + 'raw_frames\\fifa\\am_2\\') if f.endswith('.jpg') ]

for i in test_file:
    img = preprocess_images([base_dir + 'raw_frames\\fifa\\am_2\\'+i],shape_r,shape_c)
    img = torch.tensor(img.copy(),dtype=torch.float)
    for idx,x in enumerate(img):
        img[idx] = normalize(x)
    img = img.cuda()
    pred = model.forward(img)
    cv2.imwrite(base_dir + 'maps\\test\\' + i[:-3] + 'png',pred[0].squeeze(0).data.cpu().numpy())
'''
