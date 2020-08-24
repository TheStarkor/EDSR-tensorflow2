# EDSR-tensorflow2
Tensorflow 2 implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution(CVPR2017)](https://arxiv.org/abs/1707.02921)

![EDSR](https://github.com/Saafke/EDSR_Tensorflow/raw/master/images/EDSR.png?raw=true)

## Usage
```
$ python main.py [-h] N_TRAIN_DATA N_TEST_DATA N_RES_BLOCK BATCH_SIZE EPOCHS
```
### DIV2K example
```
$ python main.py 800 100 32 16 200
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
- trained by Urban100
- test image set is Set14

  
![sample](https://user-images.githubusercontent.com/45455072/91014933-40966d00-e625-11ea-801a-9f70ef231586.png)