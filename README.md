# EfficientPose
![](utils/EfficientPose.gif)

**Publicly accessible scalable single-person pose estimation as introduced in** [**"EfficientPose: Scalable single-person pose estimation"**](https://link.springer.com/article/10.1007/s10489-020-01918-7)**. We provide a simple intuitive interface for high-precision movement extraction from 2D images, videos, or directly from your webcamera.** 

**NOTE:** *All data remains safely at your computer during use.*

## Live demo

### 1. Plug

Assuming you have [Python](https://www.python.org/downloads/) (>= 3.6) and [FFMPEG](http://ffmpeg.org/download.html) (>= 4.2) preinstalled, simply run: 

```pip install -r requirements.txt```

### 2. Play

Say the magical two words:

```python track.py```

## Explore

Did I forget to mention flexibility? Indeed there is!

You are provided with these options (which go seamlessly hand in hand):
- **Path (*--path*, *-p*)**: Tell the program which file (i.e., video or image) you want to analyze. Ignore this option for camera-based tracking. For ex: ```python track.py --path=utils/MPII.jpg```

- **Model (*--model*, *-m*)**: Explore choice of model (EfficientPose RT - IV or EfficientPose RT Lite - II Lite) depending on your computational resources and precision requirements. For more details, we refer to the [performance comparison](#evidence). For ex: ```python track.py --model=II_Lite```

- **Framework (*--framework*, *-f*)**: Have specific preference of deep learning framework? We provide models in [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [TFLite](https://www.tensorflow.org/lite) and [PyTorch](https://pytorch.org/). In general, TensorFlow is recommended for maximal precision with low computational overhead on GPU, while TFLite (and PyTorch in case of ARM CPUs with [QNNPACK](https://engineering.fb.com/ml-applications/qnnpack/)) supports use in resource-constrained applications, such as smartphones. For ex: ```python track.py --framework=tensorflow```

- **Visualize predictions (*--visualize*, *-v*)**: Visualizes the keypoint predictions on top of the image/video you provided and stores the file in the folder of the original file. For ex: ```python track.py --path=utils/MPII.jpg --visualize```

- **Save predictions (*--store*, *-s*)**: Stores the predicted coordinates of 16 keypoints (top of head, upper neck, shoulders, elbows, wrists, thorax, pelvis, hips, knees, and ankles) from image/video/camera as a CSV file. Run: ```python track.py --store```

## Evidence

| Model | Resolution | Parameters | FLOPs | PCK<sub>h</sub>@50 (MPII val)| PCK<sub>h</sub>@10 (MPII val)| PCK<sub>h</sub>@50 (MPII test)| PCK<sub>h</sub>@10 (MPII test)|
| :--  | --- | --- | --- | --- | --- | --- | --- | 
| EfficientPose RT Lite<sup>*</sup> | 224x224 | 0.40M  | 0.86G | 80.6 | 23.1 | - | - | 
| EfficientPose RT | 224x224 | 0.46M  | 0.87G | 82.9 | 23.6 | 84.8 | 24.2 | 
| EfficientPose I Lite<sup>*</sup> | 256x256 | 0.59M | 1.54G | 83.7 | 27.7 | - | - |
| EfficientPose I | 256x256 | 0.72M | 1.67G | 85.2 | 26.5 | - | - |
| EfficientPose II Lite<sup>*</sup> | 368x368 | 1.46M | 7.25G | 87.1 | 30.8 | - | - |
| EfficientPose II | 368x368 | 1.73M | 7.70G | 88.2 | 30.2 | - | - |
| EfficientPose III | 480x480 | 3.23M | 23.35G | 89.5 | 30.9 | - | -  |
| EfficientPose IV | 600x600 | 6.56M | 72.89G | **89.8** | **35.6** | **91.2** | **36.0** |
| OpenPose [(Cao et al.)](https://arxiv.org/abs/1812.08008) | 368x368 | 25.94M | 160.36G | 87.6 | 22.8 | 88.8 | 22.5 |

<sup>*</sup>EfficientPose Lite models modify the original EfficientPose architecture to target deployment on edge devices through low latency inference. In particular, the EfficientNet backbones are replaced by EfficientNet Lite models of similar scaling, parallel feature extraction and cross-resolution features are omitted, squeeze-and-excitation modules are removed, and E-swish activations are replaced by ReLU6.

*All models were trained with similar optimization procedure and the precision was evaluated on the single-person [MPII benchmark](http://human-pose.mpi-inf.mpg.de/) in terms of PCK<sub>h</sub>@50 and PCK<sub>h</sub>@10. Due to restriction in number of attempts on MPII test, only EfficientPose RT and IV, and the baseline method OpenPose were officially evaluated.*

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientpose-scalable-single-person-pose/pose-estimation-on-mpii-human-pose)](https://paperswithcode.com/sota/pose-estimation-on-mpii-human-pose?p=efficientpose-scalable-single-person-pose)

## Guidelines

To achieve the optimal precision provided by the software, please adhere to the following three principles:
1. *Ensure there is only one person present in the image/video*
2. *Ensure that the full body of the person is clearly visible, is near the centre of the image/video frame, and is occupying a sufficient portion of the camera view (portrait mode is recommended when feasible)*
3. *Avoid that the subject is occluded by other objects, even partial occlusion is discouraged*

## Acknowledgment

The work is conducted as a collaboration between the [Department of Neuromedicine and Movement Science](https://www.ntnu.edu/inb) and the [Department of Computer Science](https://www.ntnu.edu/idi) at the [Norwegian University of Science and Technology](https://www.ntnu.edu/). State-of-the-art computational infrastructure is provided by the [Norwegian Open AI Lab](https://www.ntnu.edu/ailab). We are also greatful to [TensorFlow](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) and Data Scientist [Pavel Yakubovskiy](https://github.com/qubvel/efficientnet) for making EfficientNet and EfficientNet Lite models more widely available; this provided a great starting point for our research.

## Citation

If you enjoyed this project or found the work helpful in your research, please cite the following:
```
@article{groos2021efficientpose,
  title={EfficientPose: Scalable single-person pose estimation},
  author={Groos, Daniel and Ramampiaro, Heri and Ihlen, Espen AF},
  journal={Applied Intelligence},
  volume={51},
  number={4},
  pages={2518--2533},
  year={2021},
  publisher={Springer}
}
```

## Main updates

- April 2021: Provided support for TensorFlow 2
- November 2020: Launched EfficientPose Lite models in Keras, TensorFlow and TFLite
- April 2020: Launched EfficientPose models in Keras, TensorFlow, TFLite and PyTorch

## Relevant resources

- Model quantization (OpenVINO, TFLite, TensorFlow.js, TensorRT, Core ML, and ONNX): [EfficientPose in PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/084_EfficientPose)
- TensorFlow.js implementation: [EfficientPose for TFJS](https://github.com/vladmandic/efficientpose)
