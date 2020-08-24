from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse

from data_generator import train_data_generator, test_data_generator
from utils import mae, psnr
from model import edsr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N_TRAIN_DATA", type=int)
    parser.add_argument("N_TEST_DATA", type=int)
    parser.add_argument("N_RES_BLOCK", type=int)
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    parser.add_argument("SCALE", type=int)
    args = parser.parse_args()

    DATA_DIR = "../src/"
    FILE_PATH = "./models/srcnn_div2k.hdf5"
    TRAIN_PATH = "DIV2K_train_HR"
    TEST_PATH = "DIV2K_valid_HR"

    N_TRAIN_DATA = args.N_TRAIN_DATA
    N_TEST_DATA = args.N_TEST_DATA
    N_RES_BLOCK = args.N_RES_BLOCK
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    SCALE = args.SCALE

    train_data_generator = train_data_generator(
        DATA_DIR, TRAIN_PATH, scale=float(SCALE), batch_size=2
    )

    test_x, test_y = next(
        test_data_generator(DATA_DIR, TEST_PATH, scale=float(SCALE), batch_size=2, shuffle=False)
    )

    model = edsr(
        scale=SCALE, num_filters=256, num_res_blocks=N_RES_BLOCK, res_block_scaling=0.1
    )

    model.summary(line_length=150)

    lr_decay = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=10, verbose=1, min_lr=1e-5
    )
    checkpointer = ModelCheckpoint(FILE_PATH, verbose=1, save_best_only=True)
    callback_list = [lr_decay, checkpointer]

    optimizers = Adam(lr=1e-4, beta_1=0.9, beta_2=0.99)

    model.compile(loss=mae, metrics=[psnr], optimizer=optimizers)

    model.fit_generator(
        train_data_generator,
        validation_data=(test_x, test_y),
        steps_per_epoch=N_TRAIN_DATA // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callback_list,
    )
