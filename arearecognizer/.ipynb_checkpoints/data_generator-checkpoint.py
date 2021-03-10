import numpy as np
import tensorflow.keras
from tensorflow.python.keras.preprocessing.image import load_img
from utils import image_resizer as rs
import cv2


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, x_path, y_path, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, is_att=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_path = x_path
        self.y_path = y_path
        self.on_epoch_end()
        self.num_images_vis = 0
        self.is_att = is_att

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            x_img = np.asarray(load_img(self.x_path + ID), dtype=np.float32) // 255
            x_img =  x_img[:, :, :self.n_channels]
            X[i,:,:] = x_img 
            
            masked_output = rs.create_output_masked_tensor(ID, self.labels, self.dim, self.y_path, True) // 255
            y[i] = masked_output
            #self.num_images_vis += 1
            #if self.num_images_vis < 5:
            #    print(ID, self.x_path + ID)
            #    cv2.imwrite(f"data/image_orig_{self.num_images_vis}.png", x_img * 255)
            #    cv2.imwrite(f"data/image_{self.num_images_vis}.png", masked_output * 255)
        if self.is_att:
            new_x = np.empty((self.batch_size, self.n_channels, *self.dim))
            new_y = np.empty((self.batch_size, self.n_classes, *self.dim))
            for i in range(self.n_channels):
                new_x[:,i,:,:] = X[:,:,:,i]
            for i in range(self.n_classes):
                new_y[:,i,:,:] = np.asarray(y, dtype=np.float32)[:,:,:,i]
            return new_x, new_y
        return X, y#np.asarray(y, dtype=np.float64)