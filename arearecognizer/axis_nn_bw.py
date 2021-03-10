import datetime
import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.framework.ops import disable_eager_execution
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import image_resizer as rs
import arearecognizer.data_generator as dg
from arearecognizer import model_att_unet as att_model
import arearecognizer.model_unet_bw as m


#from tensorflow.python.compiler.mlcompute import mlcompute
disable_eager_execution()
#tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()
#mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and 'any'.
print(f'executing eagerly: {tf.executing_eagerly()}')


#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_memory_growth(gpus[1], True)

# import keras.backend.tensorflow_backend as tfback
# def _get_available_gpus():
#     if tfback._LOCAL_DEVICES is None:
#         devices = tf.config.list_logical_devices()
#         tfback._LOCAL_DEVICES = [x.name for x in devices]
#     return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
# tfback._get_available_gpus = _get_available_gpus
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


CLASS_NAMES = ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"]
#CLASS_NAMES = ["Title", "xpoints", "ypoints"]
base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/recognizer/imgs/")
np.random.seed(10)
seed = 42
n_channels = 3 # 1 or 3

def invoke_generator(gen, size=100):
    for i in range(size):
        gen.next()

def generate_aug_data(train_size=100, val_size=50):

    imgs = os.listdir(base_path + "resized")
    X = np.zeros((len(imgs), 512, 512, n_channels))
    Y = np.zeros((len(imgs), 512, 512, 1, len(CLASS_NAMES)))
    for i, filename in (enumerate(os.listdir(base_path + "resized/"))):
        if not filename.endswith('.png'): 
            continue
        full_path = base_path + "resized/" + filename
        x_img = cv2.imread(full_path)
        if n_channels == 1:
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
            x_img = cv2.adaptiveThreshold(x_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
            X[i, :, :, 0] = np.asarray(x_img, dtype=np.uint8)
        else:
            X[i, :, :, :n_channels] = np.asarray(x_img, dtype=np.uint8)
        Y[i] = rs.create_output_masked_tensor(filename, CLASS_NAMES)

    X_train = X[:int(X.shape[0] * 0.7)]
    X_test = X[int(X.shape[0] * 0.7):]
    Y_train = Y[:int(Y.shape[0] * 0.7)]
    Y_test = Y[int(Y.shape[0] * 0.7):]

    data_gen_args = dict(
        shear_range=0.1,
        rotation_range=30,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    image_datagen = image.ImageDataGenerator(**data_gen_args)
    mask_datagen = [image.ImageDataGenerator(**data_gen_args) for i in range(len(CLASS_NAMES))]

    image_datagen.fit(X_train[:int(X_train.shape[0] * 0.9)], augment=True, seed=seed)

    for i in range(len(CLASS_NAMES)):
        mask_datagen[i].fit(Y_train[:int(Y_train.shape[0] * 0.9), :, :, :, i], augment=True, seed=seed)

    print('starting image generation')
    batch_size_gen = 1
    x = image_datagen.flow(X_train[:int(X_train.shape[0] * 0.9)], batch_size=batch_size_gen, shuffle=True, seed=seed, save_to_dir=base_path + "aug/x/", save_format="png")
    y = [
        mask_datagen[i].flow(
            Y_train[:int(Y_train.shape[0] * 0.9), :, :, :, i],
            batch_size=batch_size_gen,
            shuffle=True,
            seed=seed,
            save_to_dir=base_path + f"aug/y/{CLASS_NAMES[i]}",
            save_format="png")
        for i in range(len(CLASS_NAMES))
    ]

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator(**data_gen_args)
    mask_datagen_val = [image.ImageDataGenerator(**data_gen_args) for i in range(len(CLASS_NAMES))]

    image_datagen_val.fit(X_train[int(X_train.shape[0] * 0.9):], augment=True, seed=seed)
    for i in range(len(CLASS_NAMES)):
        mask_datagen_val[i].fit(Y_train[int(Y_train.shape[0] * 0.9):, :, :, :, i], augment=True, seed=seed)

    x_val = image_datagen_val.flow(X_train[int(X_train.shape[0] * 0.9):], batch_size=batch_size_gen, shuffle=True,
                                   seed=seed, save_to_dir=base_path + "aug/xval/", save_format="png")
    print(Y_train.shape)
    y_val = [
        mask_datagen_val[i].flow(
            Y_train[int(Y_train.shape[0] * 0.9):, :, :, :, i],
            batch_size=batch_size_gen,
            shuffle=True,
            seed=seed,
            save_to_dir=base_path + f"aug/yval/{CLASS_NAMES[i]}",
            save_format="png")
        for i in range(len(CLASS_NAMES))
    ]

    print('remove old data and create all required dirs')
    shutil.rmtree(base_path + f"aug/x/", ignore_errors=True)
    shutil.rmtree(base_path + f"aug/xval/", ignore_errors=True)
    os.mkdir(base_path + f"aug/x")
    os.mkdir(base_path + f"aug/xval")
    for i in range(len(CLASS_NAMES)):
        if os.path.exists(base_path + f"aug/y/{CLASS_NAMES[i]}"):
            shutil.rmtree(base_path + f"aug/y/{CLASS_NAMES[i]}", ignore_errors=True)
        os.mkdir(base_path + f"aug/y/{CLASS_NAMES[i]}")
        if os.path.exists(base_path + f"aug/yval/{CLASS_NAMES[i]}"):
            shutil.rmtree(base_path + f"aug/yval/{CLASS_NAMES[i]}", ignore_errors=True)
        os.mkdir(base_path + f"aug/yval/{CLASS_NAMES[i]}")
    print('finished')

    print('generation: x')
    invoke_generator(x, train_size)
    print('generation: x_val')
    invoke_generator(x_val, val_size)
    for i in range(len(CLASS_NAMES)):
        print(f'generation: y: {CLASS_NAMES[i]}')
        invoke_generator(y[i], train_size)
        print(f'generation: y_val: {CLASS_NAMES[i]}')
        invoke_generator(y_val[i], val_size)


def train_model():
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    model = att_model.att_r2_unet(512, 512, len(CLASS_NAMES), need_compile=False, data_format="channels_first")
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    #optimizer = tf.keras.optimizers.SGD(lr=1e-7)
    metrics = [tf.keras.metrics.categorical_accuracy]
    model.compile(optimizer, loss, metrics)
    print(model.summary())

    earlystopper = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint('../trained_models/model-plots.h5', verbose=1, save_best_only=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    print('start training process')
    batch_size = 4
    params= {'dim': (512,512),
          'batch_size': batch_size,
          'n_classes': len(CLASS_NAMES),
          'n_channels': n_channels,
          'shuffle': True,
          'labels': CLASS_NAMES
    }
    x_files = [i for i in os.listdir(base_path + "aug/x/") if i.endswith(".png")]
    x_val_files = [i for i in os.listdir(base_path + "aug/xval/") if i.endswith(".png")]
    train_gen = dg.DataGenerator(x_files, x_path=base_path + "aug/x/", y_path=base_path + "aug/y/", **params, is_att=True)
    val_gen = dg.DataGenerator(x_val_files, x_path=base_path + "aug/xval/", y_path=base_path + "aug/yval/", **params, is_att=True)
    results = model.fit(train_gen,
                        validation_data=val_gen,
                        validation_steps=10,
                        steps_per_epoch=200,
                        epochs=50,
                        callbacks=[checkpointer, tensorboard_callback] #earlystopper,
                        )
     

if __name__ == "__main__":
    #generate_aug_data(5000, 1000)
    train_model()

