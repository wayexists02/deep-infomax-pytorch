import torch
import numpy as np
from settings import *


class STLDataLoader():

    def __init__(self, batch_size, mode="train", shuffle=True, noise=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise = noise
        
        self.x_path = None
        self.y_path = None

        if mode == "train":
            self.x_path = TRAIN_DATA_PATH
            self.y_path = TRAIN_LABEL_PATH
        elif mode == "valid":
            self.x_path = VALID_DATA_PATH
            self.y_path = VALID_LABEL_PATH
        elif mode == "test":
            self.x_path = TEST_DATA_PATH
        else:
            raise ValueError(f"INVALID mode: {mode}")

        self.data, self.label = self._read_data()
        self.N = self.data.shape[0]
        self.n_batches = int(np.ceil(self.N / batch_size))

        print(f"Number of data samples is {self.N}")

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle is True:
            indices = np.arange(self.N)
            np.random.shuffle(indices)

        data = self.data[indices]
        label = self.label[indices]

        for b in range(len(self)):
            start = b*self.batch_size
            end = min(self.N, (b+1)*self.batch_size)

            x_batch = self.data[start:end]
            y_batch = self.label[start:end]

            yield x_batch, y_batch

    def _read_data(self):
        with open(self.x_path, "rb") as f:
            images = np.fromfile(f, dtype=np.int8)
        with open(self.y_path, "rb") as f:
            labels = np.fromfile(f, dtype=np.uint8)

        images = images.reshape(-1, 3, 96, 96).astype(np.float32).transpose(0, 1, 3, 2)

        images = (images - 128) / 256

        images_with_noise = self._random_noise(images)
        return images_with_noise, labels

    def _random_noise(self, image):
        if self.noise is True and np.random.rand() < 0.5:
            sd = np.random.rand(*image.shape)*0.1
            image = np.random.normal(loc=image, scale=sd)
            image[image > 0.5] = 0.5
            image[image < -0.5] = -0.5

        return image
