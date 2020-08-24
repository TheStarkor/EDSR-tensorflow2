# EDSR-tensorflow2
Tensorflow 2 implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution(CVPR2017)](https://arxiv.org/abs/1707.02921).  
Most of the existing codes performed data augmentation using matlab, opencv, and numpy. This was replaced by tensorflow's ImageDataGenerator, and tensorflow 2's methods were used as a whole.

![EDSR](https://github.com/Saafke/EDSR_Tensorflow/raw/master/images/EDSR.png?raw=true)

## Usage
```
$ python main.py [-h] N_TRAIN_DATA N_TEST_DATA N_RES_BLOCK BATCH_SIZE EPOCHS SCALE
```
### DIV2K example
```
$ python main.py 800 100 32 16 200 4
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

### x2
![edsrx2](https://user-images.githubusercontent.com/45455072/91014933-40966d00-e625-11ea-801a-9f70ef231586.png)

### x4
![edsrx4](https://user-images.githubusercontent.com/45455072/91025377-45165200-e634-11ea-928f-10103a9446b8.png)