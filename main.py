from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse

from data_generator import train_data_generator, test_data_generator
from model import edsr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("TARGET_SIZE", type=int)
    parser.add_argument("N_TRAIN_DATA", type=int)
    parser.add_argument("N_TEST_DATA", type=int)
    parser.add_argument("N_RES_BLOCK", type=int)
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    parser.add_argument("SCALE", type=int)
    parser.add_argument("SAVE_NAME", type=str)
    args = parser.parse_args()

    DATA_DIR = "../src/"
    FILE_PATH = "./models/" + args.SAVE_NAME
    TRAIN_PATH = "DIV2K_train_HR"
    TEST_PATH = "DIV2K_valid_HR"

    TARGET_SIZE = args.TARGET_SIZE
    N_TRAIN_DATA = args.N_TRAIN_DATA
    N_TEST_DATA = args.N_TEST_DATA
    N_RES_BLOCK = args.N_RES_BLOCK
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    SCALE = args.SCALE

    train_data_generator = train_data_generator(
        DATA_DIR, TRAIN_PATH, scale=float(SCALE), target_size=(TARGET_SIZE, TARGET_SIZE), batch_size=2
    )

    test_x, test_y = next(
        test_data_generator(DATA_DIR, TEST_PATH, scale=float(SCALE), target_size=(TARGET_SIZE, TARGET_SIZE), batch_size=2, shuffle=False)
    )

    model = edsr(
        scale=SCALE, num_filters=256, num_res_blocks=N_RES_BLOCK, res_block_scaling=0.1
    )

    model.summary(line_length=150)

    lr_decay = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=10, verbose=1, min_lr=1e-5
    )
    checkpointer = ModelCheckpoint(FILE_PATH, verbose=1, save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir="./logs")

    callback_list = [lr_decay, checkpointer, tensorboard_callback]

    model.fit(
        train_data_generator,
        validation_data=(test_x, test_y),
        steps_per_epoch=N_TRAIN_DATA // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callback_list,
    )
