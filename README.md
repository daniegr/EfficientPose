# EfficientPose
![](utils/EfficientPose.gif)

**Publicly accessible scalable single-person pose estimation as introduced in** [**"EfficientPose: Scalable single-person pose estimation"**](https://arxiv.org/abs/****.*****)**. We provide a simple intuitive interface for high-precision movement extraction from 2D images, videos, and even the webcamera.** 

**NOTE:** *All data remains safely at your computer during use.*

## Live demo

### 1. Plug

Assuming you have [Python](https://www.python.org/downloads/) ($\geq$ 3.6) and [FFMPEG](http://ffmpeg.org/download.html) ($\geq$ 4.2) preinstalled, simply run: 

```pip install -r requirements.txt```

### 2. Play

Say the magical two words:

```python track.py```

## Explore

Did I forget to mention flexibility? Indeed there is!

You are provided with these options (which go seamlessly hand in hand):
- **Path (*--path*, *-p*)**: Tell the program which file (i.e. video or image) you want to analyze. For ex: ```python track.py --path=utils/MPII.jpg```

- **Model (*--model*, *-m*)**: Explore choice of model (EfficientPose RT-IV) depending on your computational resources and precision requirements. For more details, we refer to the [performance comparison](#performance). For ex: ```python track.py --model=IV```

- **Framework (*--framework*, *-f*)**: Have specific preference of deep learning framework? We provide models in [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [TFLite](https://www.tensorflow.org/lite) and [PyTorch](https://pytorch.org/). For ex: ```python track.py --framework=tflite```

- **Visualize predictions (*--visualize*, *-v*)**: Visualizes the keypoint predictions on top of the image/video you provided and stores the file in the folder of the original file. For ex: ```python track.py --path=utils/MPII.jpg --visualize```

- **Save predictions (*--store*, *-s*)**: Stores the predicted coordinates of 16 keypoints (top of head, upper neck, shoulders, elbows, wrists, thorax, pelvis, hips, knees, and ankles) from image/video/camera as a CSV file. Run: ```python track.py --store```

## Performance

| Model | Resolution | Parameters | FLOPs | PCK<sub>h</sub>@50 (MPII val)| PCK<sub>h</sub>@10 (MPII val)| PCK<sub>h</sub>@50 (MPII test)| PCK<sub>h</sub>@10 (MPII test)|
| :--  | --- | --- | --- | --- | --- | --- | --- | 
| EfficientPose RT | 224x224 | 0.46M  | 0.87G | 82.9 | 23.6 | 84.8 | 24.2 | 
| EfficientPose I | 256x256 | 0.72M | 1.67G | 85.2 | 26.5 | - | - |
| EfficientPose II | 368x368 | 1.73M | 7.70G | 88.2 | 30.2 | - | - |
| EfficientPose III | 480x480 | 3.23M | 23.35G | 89.5 | 30.9 | - | -  |
| EfficientPose IV | 600x600 | 6.56M | 72.89G | **89.8** | **35.6** | **91.2** | **34.0** |
| OpenPose [(Cao et al.)](https://arxiv.org/abs/1812.08008) | 368x368 | 25.94M | 160.36G | 87.6 | 22.8 | 88.8 | 22.5 |

*All models were trained with similar optimization procedure and the precision was evaluated on the single-person [MPII benchmark](http://human-pose.mpi-inf.mpg.de/) in terms of PCK<sub>h</sub>@50 and PCK<sub>h</sub>@10. Due to restriction in number of attempts on MPII test, only EfficientPose RT and IV, and the baseline method OpenPose were officially evaluated.*