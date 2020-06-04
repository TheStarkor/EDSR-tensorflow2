from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import argparse


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-dtrain",
        "--data_path",
        default="./train/lr",
        required=False,
        help="path to input image",
    )
    ap.add_argument(
        "-dlabel",
        "--label_path",
        default="./train/hr",
        required=False,
        help="path to input label",
    )
    ap.add_argument(
        "-dtest",
        "--test_path",
        default="imgs/test_data",
        required=False,
        help="path to test image",
    )
    ap.add_argument(
        "-dtlabel",
        "--tlabel_path",
        default="imgs/test_label",
        required=False,
        help="path to test label",
    )
    ap.add_argument(
        "-npath",
        "--npy_path",
        default="data/",
        required=False,
        help="path to .npy files",
    )
    ap.add_argument(
        "-itype",
        "--img_type",
        default="jpg",
        required=False,
        help="path to output model",
    )
    ap.add_argument(
        "-r",
        "--rows",
        default=320,
        required=False,
        type=int,
        help="shape of rows of input image",
    )
    ap.add_argument(
        "-c",
        "--cols",
        default=480,
        required=False,
        type=int,
        help="shape of cols of input image",
    )
    args = vars(ap.parse_args())
    return args


def create_train_data(data_path, img_type, rows, cols, label_path, npy_path):
    print("-" * 30)
    print("Creating training images...")
    print("-" * 30)

    imgs = glob.glob(data_path+"/*." + img_type)
    print(len(imgs))
    imgdatas = np.ndarray(
        (len(imgs), (int)(rows / 2), (int)(cols / 2), 3), dtype=np.uint8
    )
    imglabels = np.ndarray((len(imgs), rows, cols, 3), dtype=np.uint8)

    for i, imgname in enumerate(imgs):
        midname = imgname[imgname.rindex("/") + 1 :]
        img = load_img(data_path + "/" + midname)
        img = img_to_array(img)
        label = load_img(label_path + "/" + midname)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label

        if i % 100 == 0:
            print("Done: {0}/{1} images".format(i, len(imgs)))

    np.save(npy_path + "lr.npy", imgdatas)
    np.save(npy_path + "hr.npy", imglabels)
    print("Saving to .npy files done.")


if __name__ == "__main__":
    args = args_parse()
    data_path = args["data_path"]
    label_path = args["label_path"]
    test_path = args["test_path"]
    tlabel_path = args["tlabel_path"]
    npy_path = args["npy_path"]
    img_type = args["img_type"]
    rows = args["rows"]
    cols = args["cols"]
    create_train_data(data_path, img_type, rows, cols, label_path, npy_path)
