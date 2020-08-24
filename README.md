# EDSR-tensorflow2
Tensorflow 2 implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution(CVPR2017)](https://arxiv.org/abs/1707.02921)

![EDSR](https://github.com/Saafke/EDSR_Tensorflow/raw/master/images/EDSR.png?raw=true)

## Usage
```
$ python main.py
```
### Prerequisites
- Python 3.7
- Tensorflow 2
- Numpy

## Directory
```
EDSR-tensorflow2
├── main.py              // main program
├── model.py             // edsr model
├── data_generator.py    // data augmentation
└── utils.py             // psnr, mae
```

## Sample Results
trained by Urban100
![sample]("https://user-images.githubusercontent.com/45455072/91014943-468c4e00-e625-11ea-891e-ed2210184ba7.png")