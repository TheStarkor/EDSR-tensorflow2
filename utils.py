from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras import backend as K
import numpy as np


def mae(hr, sr):
    return mean_absolute_error(hr, sr)

def psnr(y_true,y_pred):
    return -10*K.log(K.mean(K.flatten((y_true-y_pred))**2))/np.log(10)