import os
from os import listdir
from os.path import isfile, join

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')
from keras import Model
from video import get_images

MODEL_PATH = "trained_models/vww_96.h5"
IMAGE_SIZE = 96
VIDEO_PATH = "./video_96p.mp4"
VIDEO_DATA_FOLDER = 'video_data'

vww_model = tf.keras.models.load_model(MODEL_PATH)

vww_model.summary()

# Get index of desired layer, ie conv2d_4
for idx, layer in enumerate(vww_model.layers):
    print(layer.name)
    # if layer.name == 'conv2d_4':
    #     break
    # if layer.name == 'batch_normalization_1':
    #     break
    if layer.name == 'batch_normalization_5':
        break

new_model = Model(inputs=vww_model.inputs, outputs=vww_model.layers[idx].output)

new_model.summary()

# Create datagen from video
get_images(VIDEO_PATH, VIDEO_DATA_FOLDER)
# Data loader
# files = [f for f in listdir(VIDEO_DATA_FOLDER) if isfile(join(VIDEO_DATA_FOLDER, f))]
# # Sort numerically from start to end frame
# files.sort(key=lambda x: int(x.split('.')[0]))
data = tf.keras.utils.image_dataset_from_directory(
    './' + VIDEO_DATA_FOLDER, 
    batch_size = 1,
    shuffle=False,
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
)

# vww trained model inference
# idx = 0
# for image, label in data:
#     # Should be tensor of (1, 96, 96, 3), no labels
#     print(idx)
#     idx +=1
#     out = vww_model.predict(image)
#     print(out)

# feature extraction ?
out = []
inp = []
for image, label in data:
    # Should be tensor of (1, 96, 96, 3), no labels
    # print(idx)
    idx +=1
    out.append(new_model.predict(image))
    inp.append(image)
    # print(out)
    # tf.print(out[1]-out[2], summarize=-1)
    # tf.print(out[0]-out[1], summarize=-1)
    # tf.print(inp[0]-inp[1], summarize=-1)