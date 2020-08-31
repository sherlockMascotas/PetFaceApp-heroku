import numpy as np
import json
import os
from skimage import io, transform
import logging


# ----------------------------------------------------------------------------
# Loading filenames definition
def load_dataset(directory, PATH):
    """
    Load images, labels and similarities given a directory
    returns:
    - filenames = list of filenames including root and directory
    - labels = list of labels given by directories i.e [0,0,0,1,1,2,2,2]
                has the same shape as filenames
    - similarities = dictionary with first, second and third level similarities
                    i.e {0:{'first_level':[1,2]}} indicates folder 0 has
                    first level similarity with folders 1 and 2
    """
    filenames = np.empty(0)
    labels = np.empty(0)
    idx = 0
    logging.info('Loading the {} dataset...'.format(directory))
    for root, dirs, files in os.walk(os.path.join(PATH, directory)):
        if len(files) > 1:
            for i in range(len(files)):
                files[i] = root + '/' + files[i]
            filenames = np.append(filenames, files)
            root_i = int(root.split("/")[-1])
            labels = np.append(labels, np.ones(len(files)) * root_i)
            idx += 1
    assert len(labels) != 0, '[Error] No data provided.'

    logging.info('Done loading the {} dataset...'.format(directory))

    logging.debug("Number of files in {} data: ".format(directory) + str(len(filenames)))
    nbof_classes = len(np.unique(labels))
    logging.debug('Total number of {} classes: {:d}'.format(directory, nbof_classes))

    logging.info('Loading similarities')
    json_path_s = os.path.join(PATH, directory + '_similars.json')
    with open(json_path_s, 'r') as json_file:
        similarities = json.load(json_file)
    logging.info('Done Loading similarities')
    logging.debug('Number of keys in {} similarities: {}'.format(directory, len(similarities)))

    logging.info('Loading to scale parameters')
    json_path_ts = os.path.join(PATH, directory + '_toScale.json')
    with open(json_path_ts, 'r') as json_file:
        to_scale = json.load(json_file)
    logging.info('Done to scale parameters')
    logging.debug('Number of keys in {} to scale: {}'.format(directory, len(to_scale)))

    return filenames, labels, similarities, to_scale


# ----------------------------------------------------------------------------
# Loading images definition
def load_images(filenames, SIZE, EFFNET=False):
    """
    Use scikit-image library to load the pictures from files to numpy array.
    """
    h, w, c = SIZE
    images = np.empty((len(filenames), h, w, c))
    factor = 1 if EFFNET else 255.0
    for i, f in enumerate(filenames):
        images[i] = io.imread(f) / factor
    return images
#----------------------------------------------------------------------------
# Downscale and rescale back images
def down_up_scale(img, to_scale, SIZE = (224,224,3), factor=0.65, rand_thres=0.5):
    # randomly select if downscale and rescale back
    rand = np.random.rand()
    if rand < rand_thres and to_scale:
        new_img = transform.resize(
            transform.rescale(
                img.astype('uint8'), (factor,factor,1), anti_aliasing=False)
            , SIZE, anti_aliasing=False
        )*255
    else: new_img = img
    return new_img.astype('uint8')

# ----------------------------------------------------------------------------
# Apply and transform images
def apply_transform(images, datagen, to_scale, SIZE):
    """
    Apply a data preprocessing transformation to n images
    Args:
        -images
        -ImageDataGenerator
        -to_scale: boolean list same size as images
    Return:
        -images of the same shape of the inputs but transformed
    """
    batch_size = len(images)
    for x in datagen.flow(images, batch_size=batch_size, shuffle=False):
        for i in range(0,batch_size):
            if to_scale[i]:
                x[i] = down_up_scale(x[i], to_scale[i], SIZE)
            else:
                x[i] = x[i].astype('uint8')
        return x