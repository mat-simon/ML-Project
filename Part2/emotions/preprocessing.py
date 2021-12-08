import os
import numpy as np
import cv2


def jpg_to_numpy():
    train_path = "data/train"
    train_x = []
    train_y = []
    for emotion_folder in os.listdir(train_path):
        images_path = f"data/train/{emotion_folder}"
        # uncomment to filer class
        # if emotion_folder == "disgust":
        #     continue
        for emotion_image in os.listdir(images_path):
            image_path = images_path + f"/{emotion_image}"
            # get numpy array image in gray scale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # perform the resizing
            res = cv2.resize(img, dsize=(20, 20),
                             interpolation=cv2.INTER_NEAREST)
            data = np.asarray(res).reshape(400,)
            train_x.append(data)
            train_y.append(emotion_folder)

    # filename
    with open('data/training_x.npy', 'wb') as f:
        np.save(f, train_x)
    with open('data/training_y.npy', 'wb') as f:
        np.save(f, train_y)


if __name__ == '__main__':
    jpg_to_numpy()
