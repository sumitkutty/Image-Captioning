import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_captions(img_dict):
    """Returns the dictionary with processed captions
    Preprocessing steps:
    1) Convert all words to lower case
    2) Remove all punctuation
    3) Remove all words that have one character
    4) Remove all words with numbers

    Args:
        img_dict ([Dictionary]): dictionary in form of {img_name :[caption1, caption2, ....]}

    Returns:
        [Dictionary]: Dictionary with preprocessed captions
    """   

    for img_name, captions in img_dict.items():
        for i, caption in enumerate(captions):
            caption_nopunc = re.sub(r"[^A-Za-z0-9]", " ", caption.lower())
            clean_words = [word for word in caption_nopunc.split() if len(word)>1 and word.isalpha()]

            new_caption = ' '.join(clean_words)

            captions[i] = new_caption


    return img_dict



def add_sof_eof(captions):
    """Adds indicators for start of caption and end of caption

    Args:
        captions ([List]): [List of captions for a single image]

    Returns:
        [List]: [List of captions for a single image after adding indicators]
    """    
    for i, caption in enumerate(captions):
        captions[i] = 'startseq ' + caption + ' endseq'

    return captions

def add_token(img_dict, img_names):
    """Function to facilitate adding indicators for start and end of captions

    Args:
        img_dict ([Dict]): dictionary in form of {img_name :[caption1, caption2, ....]}
        img_names ([List]): list of all image names in the train set

    Returns:
        [Dict]: Dictionary of form {img_name: [caption1, caption2, ..] with added indicators for captions}
    """    

    img_dict = {img_name: add_sof_eof(captions) for img_name, captions in img_dict.items() if img_name in img_names}
    return img_dict





def create_tokenizer(data_dict):
    """Creates a vocabulary for the captions

    Args:
        data_dict ([dict]): [Dictionary of format {img_name: [caption1, caption2, ..]}]

    Returns:
        [Keras Tokenizer Object]: [description]
        [int]: Length of the vocabulary
        [int]: length of the caption with most words
    """    
    # Store all the captions of all images (to make a volcabulary)
    captions = [caption for key, captions in data_dict.items() for caption in captions]

    #Length of caption with most words
    caption_max_len = max(len(caption.split()) for caption in captions)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)

    vocab_size = len(tokenizer.word_index) + 1

    return tokenizer, vocab_size, caption_max_len





def pad_text(text, max_length):
    """pads 0's at the end of the vectors to have same sizes

    Args:
        text ([List]): [A single vectorized caption in sequence (List of ints)]
        max_length ([int]): [Length of the caption with most words]

    Returns:
        [List]: [Padded caption veector]
    """    
    return pad_sequences([text], maxlen = max_length, padding = 'post')[0]




def prepare_training_data(data_dict, tokenizer, max_length, vocab_size, image_files):
    """vectorizes and pads the captions 

    Args:
        data_dict ([dict]): [Dictionary of form -> {img_name:[caption1, caption2, ..]}]
        tokenizer ([Keras Tokenizer]): [Tokenizer used to create sequences]
        max_length ([int]): [Length of the caption with most words]
        vocab_size ([int]): [Size of the vocab]
        image_files (str):  [Path to all the image files]

    Returns:
        x ([array]): [Array with all train image paths]
        y ([array]): [Array with encoded vectorized captions for each image]
    """    
    x , y = list() ,list()
    for img_name, captions in data_dict.items():
        img_path = image_files + img_name + '.jpg'

        for caption in captions:
            #converts the text sentences to sequences of numbers where the nums are the word's index in vocab
            words_ids = tokenizer.texts_to_sequences([caption])[0] 
            
            #Makes all words_ids vector of same length by padding 0's at the end(padding='post') of the vector
            padded_ids = pad_text(words_ids,max_length)

            x.append(img_path)
            y.append(padded_ids)
            
    return np.array(x), np.array(y)