from PIL import Image
import glob, os
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import json, pickle
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from utils import *

def populate_img_arr(images_paths, size=(32, 32), should_preprocess=False):
    """
    Get an array of images for a list of image paths
    Args:
        size: the size of image , in pixels
        should_preprocess: if the images should be processed (according to InceptionV3 requirements)
    Returns:
        arr: An array of the loaded images
    """
    arr = []
    for i, img_path in enumerate(images_paths):
        img = image.load_img(img_path, target_size=size)
        x = image.img_to_array(img)
        arr.append(x)
    arr = np.array(arr)
    if should_preprocess:
        arr = preprocess_input(arr)
    return arr

def images_to_sprite(data):
    """
    Creates the sprite image along with any necessary padding
    Source : https://github.com/tensorflow/tensorflow/issues/6322
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    print(data.shape)
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def main(_):
    X_train, _ = getData()
    X_test, _ = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    final_result = X_test#np.vstack((X_train, X_test))

    y = tf.Variable(final_result,name='face')
    summary_writer = tf.summary.FileWriter('logs')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name
    embedding.metadata_path = 'meta.tsv'
    embedding.sprite.image_path = 'sprite.png'
    embedding.sprite.single_image_dim.extend([32,32])
    projector.visualize_embeddings(summary_writer,config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess,os.path.join('./logs', 'model'), 10000)
    summary_writer.close()

if __name__ == '__main__':
    # with open('./data/name.pickle', 'rb') as f:
    #     train_name, test_name = pickle.load(f)
    # name = test_name
    # X_test, y_test = getData(type='test')
    # # print(np.count_nonzero(y_test == 1), len(list(y_test)))
    # print("Populating")
    # X = populate_img_arr(name)
    # print("spriting")
    # sprite = Image.fromarray(images_to_sprite(X).astype(np.uint8))
    # sprite.save('./logs/sprite.png')
    #
    # with open('./logs/meta.tsv','w') as f:
    #     f.write('Index\tLabel\n')
    #     for index, _ in enumerate(name):
    #         if index < 2017:
    #             label = 1
    #         else:
    #             label = 0
    #         f.write('{}\t{}\n'.format(index,label))

    tf.app.run()
