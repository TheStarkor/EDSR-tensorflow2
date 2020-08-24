from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from data_generator import train_data_generator, test_data_generator
from utils import mae, psnr
from model import edsr

DATA_DIR = "./"
N_TRAIN_DATA = 14
N_TEST_DATA = 5
BATCH_SIZE = 3

model = edsr(scale=4, num_filters=256, num_res_blocks=32, res_block_scaling=0.1)

model.summary(line_length=150)

lr_decay = ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=10, verbose=1, min_lr=1e-5
)
checkpointer = ModelCheckpoint("edsr.h5", verbose=1, save_best_only=True)
callback_list = [lr_decay, checkpointer]

optimizers = Adam(lr=1e-4, beta_1=0.9, beta_2=0.99)

model.compile(loss=mae, metrics=[psnr], optimizer=optimizers)

train_data_generator = train_data_generator(DATA_DIR, "train", scale=4.0, batch_size=2)

test_x, test_y = next(
    test_data_generator(DATA_DIR, "test", scale=4.0, batch_size=2, shuffle=False)
)

model.fit_generator(
    train_data_generator,
    validation_data=(test_x, test_y),
    steps_per_epoch=N_TRAIN_DATA // 2,
    epochs=100,
    callbacks=callback_list,
)
