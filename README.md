# DeepGame
Efficient Video Encoding for Cloud Gaming

The following repository includes the source code for DeepGame.

## Abstract
Cloud gaming enables users to play games on virtually any device. This is achieved by offloading the game rendering and encoding to cloud datacenters. As game resolutions and frame rates increase, cloud gaming platforms face a major challenge to stream high quality games due to the high bandwidth and low latency requirements. In this paper, we propose a new video encoding pipeline, called DeepGame, for cloud gaming platforms to reduce the bandwidth requirements with limited to no impact on the player quality of experience. DeepGame learns the player’s contextual interest in the game and the temporal correlation of that interest using a spatiotemporal deep neural network. Then, it encodes various areas in the video frames with different quality levels proportional to their contextual importance. DeepGame does not change the source code of the video encoder or the video game, and it does not require any additional hardware or software at the client side. We implemented DeepGame in an open-source cloud gaming platform and evaluated its performance using multiple popular games. We also conducted a subjective study with real players to demonstrate the potential gains achieved by DeepGame and its practicality. Our results show that DeepGame can reduce the bandwidth requirements by up to 36% compared to the baseline encoder, while maintaining the same level of perceived quality for players and running in real time.

## Dataset
First, you need to download the dataset. 
The dataset ...


## Data preparation
Next, you need to prepare the data for the machine learning algorithm. You need to preprocess the data into ...


## Analysis
Additionally, you can analyze the data to identify the number of invalid fixations along with the number of consecutive fixations.

## ROI prediction model
The ROI prediction model uses an object detection module followed by a spatio-temporal network (LSTM) to identify the ROIs withing game frames. 

### Object Detection (YOLO)
First, you need to train the object detection network on the most important game objects
A number of pre-trained models is provided. 
Next, you will use these pre-trained models to extract the objects from the game frames which will be used as inputs to the LSTM network.

### Training
The lstm is trained ...

### Testing
The inference is performed ..

## Encoding evaluation


## GamingAnywhere integration


## Citation

```
@InProceedings{10.1145/3394171.3413905,
    author = {O. Mossad, K. Diab, I. Amer, and M. Hefeeda},
    title = {DeepGame: Efficient Video Encoding for Cloud Gaming},
    year = {2021},
    isbn = {978-1-4503-8651-7/21/10},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3474085.3475594},
    doi = {10.1145/3474085.3475594},
    booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
    numpages = {9},
    keywords = {Cloud Gaming, Context-Based Video Encoding},
    location = {Chengdu, China},
    series = {MM '21}
    }
```