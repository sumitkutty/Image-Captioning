import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_caption_data(path):
    with open(path, 'r') as f:
        text = f.read()
        f.close()
        return text


def create_caption_dict(text):
    """Dictionary with image name and captions for each image

    Args:
        text ([str]): [text consisting of image names and 5 captions for every image]

    Returns:
        [Dict]: [dictionary with image names as key and list of captions for the image as value.]
    """    
    dicto = {}
    lines = text.split('\n')
    for line in lines:
        line_split = line.split('\t')

        if len(line_split) != 2:
            continue
        else:
            image_file, caption = line_split

            img_file, caption_id = image_file.split('#')

            image_name = img_file.split('.')[0]

            if int(caption_id) == 0:
                dicto[image_name] = [caption]
            else:
                dicto[image_name].append(caption)


    return dicto
    


def load_train_image_names(path):
    """Stores the names of all files from the train set

    Args:
        path (str): [path to the text file with file names of train set images]

    Returns:
        [list]: [image names for train set ]
    """    
    with open(path, 'r') as f:
        lines = f.read()
        f.close()

    file_names = lines.split('\n')
    
    names = [name.split('.')[0] for name in file_names if len(name) >= 1]
    

    return names
    



def load_image(image_path):
    """Loads image using tf

    Args:
        image_path ([str]): [Path to the image]

    Returns:
        [Tensor],[str]: [preprocessed tensor Image,  image path]
    """    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (288,288))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

    