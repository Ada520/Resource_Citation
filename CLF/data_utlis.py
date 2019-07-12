import codecs
import random
import gensim
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import nltk
import pickle

PAD_ID=0
UNK_ID=1
CITE_ID=2
_PAD="PAD"
_UNK="UNK"
_CITE="_CITE_"

def load_data(data_path,vocab_role_1st_label2index,
              vocab_role_2nd_label2index,vocab_func_label_2index,
              vocab_word2index,vocab_cahr2index,vocab_pos2index,
              vocab_cap2index,sentence_len,word_len,flag_use_char,
              flag_use_pos,flag_use_cap):
    """
    :param data_path:
    :param vocab_role_1st_label2index:
    :param vocab_role_2nd_label2index:
    :param vocab_func_label_2index:
    :param vocab_word2index:
    :param vocab_cahr2index:
    :param vocab_pos2index:
    :param vocab_cap2index:
    :param sentence_len:
    :param word_len:
    :param flag_use_char:
    :param flag_use_pos:
    :param flag_use_cap:
    :return:X: [ word_sequence, char_sequence, pos_sequence, cap_sequence]
                - word_sequence: sentence_len
                - char_sequence: sentence_len * word_len
                - pos_sequence: sentence_len
                - cap_sequence: sentence_len
    """























































