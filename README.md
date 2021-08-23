# DeepGame
Efficient Video Encoding for Cloud Gaming

## Dataset


## Data preparation


## Analysis


## ROI prediction model
### Training

### Testing

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
    abstract = {Cloud gaming enables users to play games on virtually any device. This is achieved by offloading the game rendering and encoding to cloud datacenters. As game resolutions and frame rates increase, cloud gaming platforms face a major challenge to stream high quality games due to the high bandwidth and low latency requirements. In this paper, we propose a new video encoding pipeline, called DeepGame, for cloud gaming platforms to reduce the bandwidth requirements with limited to no impact on the player quality of experience. DeepGame learns the player’s contextual interest in the game and the temporal correlation of that interest using a spatiotemporal deep neural network. Then, it encodes various areas in the video frames with different quality levels proportional to their contextual importance. DeepGame does not change the source code of the video encoder or the video game, and it does not require any additional hardware or software at the client side. We implemented DeepGame in an open-source cloud gaming platform and evaluated its performance using multiple popular games. We also conducted a subjective study with real players to demonstrate the potential gains achieved by DeepGame and its practicality. Our results show that DeepGame can reduce the bandwidth requirements by up to 36% compared to the baseline encoder, while maintaining the same level of perceived quality for players and running in real time.
},
    booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
    numpages = {9},
    keywords = {Cloud Gaming, Context-Based Video Encoding},
    location = {Chengdu, China},
    series = {MM '21}
    }
```