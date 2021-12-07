import os
from PIL import Image
import numpy as np


def jpg_to_numpy():
    train_path = "../../data/emotions/train"
    train_x = []
    train_y = []
    for emotion_folder in os.listdir(train_path):
        images_path = f"../../data/emotions/train/{emotion_folder}"
        if emotion_folder == "disgust":
            continue
        for emotion_image in os.listdir(images_path):
            image_path = images_path + f"/{emotion_image}"
            img = Image.open(image_path)
            data = np.asarray(img).reshape(2304,)
            train_x.append(data)
            train_y.append(emotion_folder)

    with open('train_x.npy', 'wb') as f:
        np.save(f, train_x)
    with open('train_y.npy', 'wb') as f:
        np.save(f, train_y)


if __name__ == '__main__':
    jpg_to_numpy()
