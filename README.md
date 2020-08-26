# EDSR-tensorflow2
Tensorflow 2 implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution(CVPR2017)](https://arxiv.org/abs/1707.02921).  
Most of the existing codes performed data augmentation using matlab, opencv, and numpy. This was replaced by tensorflow's ImageDataGenerator, and tensorflow 2's methods were used as a whole.

![EDSR](https://github.com/Saafke/EDSR_Tensorflow/raw/master/images/EDSR.png?raw=true)

## Usage
```
$ mkdir models
$ main.py [-h] N_TRAIN_DATA N_TEST_DATA N_RES_BLOCK BATCH_SIZE EPOCHS SCALE SAVE_NAME
```
### DIV2K example
```
$ mkdir models
$ python main.py 800 100 32 16 400 4 edsr_div2k.hdf5
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
- trained by DIV2K
- test image set is Set14
![edsrx4](https://user-images.githubusercontent.com/45455072/91146276-a05a4a00-e6f1-11ea-8d36-3c1235a842d8.png)
  
![psnr](https://user-images.githubusercontent.com/45455072/91146286-a6e8c180-e6f1-11ea-85c0-b4f45d953f6e.png)
  
![loss](https://user-images.githubusercontent.com/45455072/91146297-a819ee80-e6f1-11ea-9791-1e4e5c07a52c.png)