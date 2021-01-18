import cv2
import numpy as np
path = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\fifa\\ha_7\\frame_00249.jpg'
image = cv2.imread(path)
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
print(saliencyMap)
print(saliencyMap.shape)
#print(max(saliencyMap))
#print(min(saliencyMap))
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
print(np.max(saliencyMap))
print(np.min(saliencyMap))
cv2.waitKey(0)
