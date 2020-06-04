import numpy as np
from tensorflow.keras.models import load_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv
import tensorflow as tf
from edsr import edsr
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path="results"

def mae(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return mean_absolute_error(hr, sr)

def psnr(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return tf.image.psnr(hr, sr, max_val=255)

def _crop_hr_in_training(hr, sr):
    """
    Remove margin of size scale*2 from hr in training phase.
    The margin is computed from size difference of hr and sr
    so that no explicit scale parameter is needed. This is only
    needed for WDSR models.

    """
    margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
    # crop only if margin > 0
    hr_crop = tf.cond(tf.equal(margin, 0),
                      lambda: hr,
                      lambda: hr[:, margin:-margin, margin:-margin, :])
    hr = K.in_train_phase(hr_crop, hr)
    hr.uses_learning_phase = True
    return hr, sr


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", default="data/",
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", default="model_save/",
                    help="path to save the output model")
    ap.add_argument("-name","--model_name", default="edsr.h5",
                    help="output of model name")
    args = vars(ap.parse_args())
    return args


def test(args):
    X_train = np.load(args["npy_path"] + 'lr.npy')
    #X_test = np.load(args["npy_path"] + 'X_test.npy')
    #X_val = np.load(args["npy_path"] + 'X_val_ori.npy')
    y_train = np.load(args["npy_path"] + 'hr.npy')
    #y_test = np.load(args["npy_path"] + 'y_test.npy')
    #y_val = np.load(args["npy_path"] + 'y_val_concat.npy')

    model = load_model("edsr.h5",
                       custom_objects={'loss': mae, 'psnr': psnr, 'tf': tf })
    
    # test all images from test.npy     
    print(len(X_train))
    for i in range(10):
        y_pred = model.predict(X_train[i].reshape((-1, 160, 240, 3)))
        y_pred = y_pred.reshape((320, 480,3))
        y_pred = cv.normalize(y_pred, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        y_pred = cv.cvtColor(y_pred, cv.COLOR_BGR2RGB)
        cv.imwrite(path+'/'+'pred_'+format(str(i))+ '.jpg',y_pred)
        
        plt.figure(figsize=(25, 16))
        plt.subplot(1, 3, 1)
        plt.title('bicubic')
        X_bicubic=cv.resize(X_train[i], (480, 320), interpolation = cv.INTER_CUBIC)
        plt.imshow(X_bicubic, cmap='binary')
        X_bicubic = cv.cvtColor(X_bicubic, cv.COLOR_BGR2RGB)
        cv.imwrite(path+'/'+'bicubic_'+format(str(i))+ '.jpg',X_bicubic)
        
        plt.subplot(1, 3, 2)
        plt.title('reference')
        print(y_train[i].shape)        
        plt.imshow(y_train[i], cmap='binary')
        ref = cv.cvtColor(y_train[i],cv.COLOR_BGR2RGB)
        cv.imwrite(path+'/'+'ref_'+format(str(i))+ '.jpg',ref)
        
        y_pred = cv.cvtColor(y_pred, cv.COLOR_RGB2BGR)
        plt.subplot(1, 3, 3)
        plt.title('EDSR')
        plt.imshow(y_pred, cmap='binary')
        name = path+'/'+str(i) +'.jpg'       
        plt.savefig(name)
        plt.close()



if __name__ == "__main__":
    args = args_parse()
    test(args)