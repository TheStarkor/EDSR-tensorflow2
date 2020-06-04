import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras import backend as K
import argparse
import os
import tensorflow as tf
from edsr import edsr

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    hr_crop = tf.cond(
        tf.equal(margin, 0),
        lambda: hr,
        lambda: hr[:, margin:-margin, margin:-margin, :],
    )
    hr = K.in_train_phase(hr_crop, hr)
    hr.uses_learning_phase = True
    return hr, sr


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="Keras Training")
    # ========= paths for training
    ap.add_argument(
        "-npath",
        "--npy_path",
        default="data/",
        required=False,
        help="path to npy. files to train",
    )
    ap.add_argument(
        "-mpath",
        "--model_path",
        default="model_save/",
        required=False,
        help="path to save the output model",
    )
    ap.add_argument(
        "-lpath",
        "--log_path",
        default="log/",
        required=False,
        help="path to save the 'log' files",
    )
    ap.add_argument(
        "-name",
        "--model_name",
        default="edsr.h5",
        required=False,
        help="output of model name",
    )
    # ========= parameters for training
    ap.add_argument(
        "-p",
        "--pretrain",
        default=0,
        required=False,
        type=int,
        help="load pre-train model or not",
    )

    ap.add_argument("-bs", "--batch_size", default=2, type=int, help="batch size")
    ap.add_argument("-ep", "--epoch", default=10, type=int, help="epoch")
    ap.add_argument(
        "-m", "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    args = vars(ap.parse_args())
    return args


def train(args):
    scale = 2
    X_train = np.load(args["npy_path"] + "lr.npy")
    y_train = np.load(args["npy_path"] + "hr.npy")

    if args["pretrain"]:
        model = load_model(
            "edsr.h5",
            custom_objects={"loss": mae, "psnr": psnr},
        )
    else:
        model = edsr(
            scale, num_filters = 256, num_res_blocks=32, res_block_scaling=0.1
        )

    model.summary()

    lr_decay = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=10, verbose=1, min_lr=1e-5
    )
    checkpointer = ModelCheckpoint(
        "edsr.h5", verbose=1, save_best_only=True
    )
    tensorboard = TensorBoard(log_dir=args["log_path"])
    callback_list = [lr_decay, checkpointer, tensorboard]

    optimizers = Adam(lr=1e-4, beta_1=0.9, beta_2=0.99)

    model.compile(loss=mae, metrics=[psnr], optimizer=optimizers)

    EDSR = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=args["batch_size"],
        epochs=args["epoch"],
        callbacks=callback_list,
        verbose=2,
    )


if __name__ == "__main__":
    args = args_parse()
    train(args)
