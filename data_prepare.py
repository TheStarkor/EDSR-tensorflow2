import cv2
import os
import random

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 480
CUT_IMAGE_NUM = 10
FACTOR = 1 / 2


class DataProcessor(object):
    def cut_for_train(self, from_path, dest_path):
        if not os.path.exists(from_path):
            return None
        if not os.path.exists(dest_path):
            try:
                os.mkdir(dest_path)
            except:
                return None

        file_list = os.listdir(from_path)
        for i, file in enumerate(file_list):
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            if image is None:
                continue

            img_width, img_height, img_channel = image.shape

            if img_width > IMAGE_WIDTH and img_height > IMAGE_HEIGHT:
                for j in range(CUT_IMAGE_NUM):
                    m = random.randint(0, img_width - IMAGE_WIDTH)
                    n = random.randint(0, img_height - IMAGE_HEIGHT)
                    new_image = image[m : m + IMAGE_WIDTH, n : n + IMAGE_HEIGHT]

                    cv2.imwrite(
                        os.path.join(dest_path, str(i) + "_" + str(j) + ".jpg"),
                        new_image,
                    )

    def down_sampling_for_train(self, from_path, dest_path):
        if not os.path.exists(from_path):
            return None

        if not os.path.exists(dest_path):
            try:
                os.mkdir(dest_path)
            except:
                return None

        files = os.listdir(from_path)

        for file in files:
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            width, height, _ = image.shape

            image = cv2.resize(
                image,
                (int(height * FACTOR), int(width * FACTOR)),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(os.path.join(dest_path, file), image)


if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.cut_for_train("./images/", "./train/hr/")
    data_processor.down_sampling_for_train("./train/hr/", "./train/lr/")
