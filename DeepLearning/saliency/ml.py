# FILES TO EXCLUDE: pa_4, pu_0, pu_5, pu_6, pu_7
import torch
import torchvision.models as models
import utils
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt


### LOAD PARAMS
[ipt_len, opt_len] = utils.get_model_input_params()

### LOAD DATA



### MODEL
resnet50 = models.resnet50(pretrained=True)
# REMOVE LAST LAYER #
# source: https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/45 #
model = nn.Sequential(*list(resnet50.children())[:-1])

print(resnet50)
print(model)


# FIFA
sample_img = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\fifa\\ha_3\\frame_00130.jpg'
# CS:GO
sample_img = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\csgo\\ha_10\\frame_01271.jpg'
# NHL
sample_img = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\nhl\\ha_11\\frame_00130.jpg'
# NBA
#sample_img = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\nba\\ha_14\\frame_00130.jpg'


#img = cv2.imread(sample_img)
#print(img.shape)


## SALIENCY MAPS
# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
image = cv2.imread(sample_img)
print(image.shape)
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# show the images
'''
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)
'''

##

img = Image.open(sample_img)
preprocess = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #transforms.Normalize(mean=[0.385, 0.356, 0.306], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img)
plt.imshow(  input_tensor.permute(1, 2, 0)  )
plt.show()
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print(input_tensor.shape)

x = torch.randn(1, 3, 224, 224)
out = model(input_batch)
print(out)
