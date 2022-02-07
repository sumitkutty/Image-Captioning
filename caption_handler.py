import re
from tensorflow.keras.preprocessing.text import Tokenizer

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