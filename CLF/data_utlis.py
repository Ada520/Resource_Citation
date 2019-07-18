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


def word_capitalize(word):
    if word.isupper():
        return 'UPPER'
    if word.islower():
        return 'LOWER'
    if word.istitle():
        return 'TITLE'
    else:
        if word.isdigit():
            return 'NUM'
        if word.isalpha():
            return 'ALPHA'
        else:
            return 'OTHER'

def load_data(data_path,vocab_role_1st_label2index,
              vocab_role_2nd_label2index,vocab_func_label2index,
              vocab_word2index,vocab_char2index,vocab_pos2index,
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
    data_file = codecs.open(data_path, mode='r', encoding='utf-8')
    data_lines = data_file.readlines()
    random.shuffle(data_lines)
    # build data samples:
    link_Y = []
    Word_sequences = []
    Char_sequences = []
    Pos_sequences = []
    Cap_sequences = []
    role_1st_labels = []
    role_2nd_labels = []
    func_labels = []
    for i, line in enumerate(data_lines):
        #raw_list = line.strip().split("\t")
        #print("====="*15)
        #print(raw_list)
        #print(raw_list[0])
        #print(raw_list[1])
        link_index, raw_list = int(i), line
        raw_list = raw_list.strip().split("__label__")
        input_list = raw_list[0].strip().split(" ")
        label_list = raw_list[1].split('|')
        # get labels
        link_Y.append(link_index)
        role_1st_label = vocab_role_1st_label2index[label_list[0]]
        # print("====="*15)
        # print(label_list[0])
        # print(role_1st_label)
        # exit()
        role_2nd_label = vocab_role_2nd_label2index[label_list[1]]
        func_label = vocab_func_label2index[label_list[2]]
        role_1st_labels.append(role_1st_label)
        role_2nd_labels.append(role_2nd_label)
        func_labels.append(func_label)
        # get word lists
        word_sequence = [vocab_word2index.get(x, UNK_ID) for x in input_list]
        #print(word_sequence)
        #exit()
        Word_sequences.append(word_sequence)
        # get char lists
        if flag_use_char:
            char_sequence = [] # [sentence_len, word_len]
            for word in input_list:
                char_indexs = [vocab_char2index.get(char, UNK_ID) for char in word]
                char_sequence.append(char_indexs)
            if len(input_list) < sentence_len:
                char_sequence.extend( [[0]] * (sentence_len-len(input_list)))
            else:
                char_sequence = char_sequence[:sentence_len]
            #print(input_list)
            #print(char_sequence)
            char_sequence = pad_sequences(char_sequence, maxlen=word_len, value=0.)
            #print(char_sequence[0])
            #print(char_sequence)
            #exit()
            Char_sequences.append(char_sequence)
        if flag_use_pos:
            pos_sequence = nltk.pos_tag(input_list) # [sentence_len]
            word_seq, pos_seq = zip(*pos_sequence)
            pos_sequence = list(pos_seq)
            pos_sequence = [vocab_pos2index.get(pos, UNK_ID) for pos in pos_sequence]
            Pos_sequences.append(pos_sequence)
        if flag_use_cap:
            cap_sequence = [word_capitalize(word) for word in input_list]
            cap_sequence = [vocab_cap2index[cap] for cap in cap_sequence]
            Cap_sequences.append(cap_sequence)

    Word_sequences = pad_sequences(Word_sequences, maxlen=sentence_len, value=0.)
    #print(Word_sequences)
    #exit()
    if flag_use_pos:
        Pos_sequences = pad_sequences(Pos_sequences, maxlen=sentence_len, value=0.)
    if flag_use_cap:
        Cap_sequences = pad_sequences(Cap_sequences, maxlen=sentence_len, value=0.)
    X = {'word':np.array(Word_sequences), 'char':np.array(Char_sequences), 'pos':np.array(Pos_sequences), 'cap':np.array(Cap_sequences),
         'role_1st':role_1st_labels, 'role_2nd':role_2nd_labels, 'func':func_labels}
    return (X, np.array(link_Y))




#use pretrained word embeding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,word2vec_model,name_scope="att_lstm"):
    cache_vocabulary_label_pik='cache'+'_'+name_scope
    if not os.path.isdir(cache_vocabulary_label_pik):
        os.makedirs(cache_vocabulary_label_pik)
    cache_path=cache_vocabulary_label_pik+'/'+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        #pass
        with open(cache_path,'rb') as data_f:
            return pickle.load(data_f)
    else:
        #initialize vocabulary
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK
        vocabulary_word2index[_CITE]=CITE_ID
        vocabulary_index2word[CITE_ID]=_CITE

        vocabulary_role_1st_label2index={}
        vocabulary_role_1st_index2label={}
        vocabulary_role_2nd_label2index={}
        vocabulary_role_2nd_index2label={}
        vocabulary_func_label2index={}
        vocabulary_func_index2label={}

        vocabulary_char2index = {}
        vocabulary_index2char = {}
        vocabulary_char2index[_PAD]=PAD_ID
        vocabulary_index2char[PAD_ID]=_PAD
        vocabulary_char2index[_UNK]=UNK_ID
        vocabulary_index2char[UNK_ID]=_UNK

        vocabulary_pos2index = {}
        vocabulary_index2pos = {}
        vocabulary_pos2index[_PAD]=PAD_ID
        vocabulary_index2pos[PAD_ID]=_PAD
        vocabulary_pos2index[_UNK]=UNK_ID
        vocabulary_index2pos[UNK_ID]=_UNK

        vocabulary_cap2index = {}
        vocabulary_index2cap = {}
        vocabulary_cap2index[_PAD]=PAD_ID
        vocabulary_index2cap[PAD_ID]=_PAD

        #1.load raw data
        train_file=codecs.open(training_data_path,mode='r',encoding='utf-8')
        train_lines=train_file.readlines()

        #2.loop each line, put to counter
        c_words=Counter()
        c_chars=Counter()
        c_pos=Counter()
        c_cap = Counter()
        c_role_1st_labels=Counter()
        c_role_2nd_labels=Counter()
        c_func_labels=Counter()

        for line in train_lines:
            raw_list=line.strip().split("__label__")
            input_list=raw_list[0].strip().split(" ")
            label_list=raw_list[1].strip().split("|")
            c_words.update(input_list)
            #print('c_words',c_words)
            c_role_1st_labels.update([label_list[0]])
            c_role_2nd_labels.update([label_list[1]])
            c_func_labels.update([label_list[2]])
            pos_list=nltk.pos_tag(input_list)
            word_seg,pos_seg=zip(*pos_list)
            pos_list=list(pos_seg)
            c_pos.update(pos_list)
            cap_list=[word_capitalize(word) for word in input_list]
            c_cap.update(cap_list)
            for word in input_list:
                c_chars.update(word)

        #return most frequency
        if word2vec_model!=None:
            word_vocab_list=[]
            for word in word2vec_model.vocab:
                word_vocab_list.append(word)
        else:
            word_vocab_list=c_words.most_common()
        #print("======================")
        #print (word_vocab_list)
        role_1st_label_list=c_role_1st_labels.most_common()
        #print("========="*15)
        #print(role_1st_label_list)
        role_2nd_label_list=c_role_2nd_labels.most_common()
        func_label_list=c_func_labels.most_common()
        char_vocab_list = c_chars.most_common()
        pos_vocab_list = c_pos.most_common()
        cap_vocab_list = c_cap.most_common()
        #print("========="*15)
        #print(cap_vocab_list)
        #exit()
        #return most frequency

        #put those words to dict
        for i,tuplee in enumerate(word_vocab_list):
            if word2vec_model != None:
                word = tuplee
            else:
                word, _ = tuplee
            vocabulary_word2index[word] = i+3
            vocabulary_index2word[i+3] = word

        for i,tuplee in enumerate(char_vocab_list):
            char, _ = tuplee
            vocabulary_char2index[char] = i+2
            vocabulary_index2char[i+2] = char

        for i,tuplee in enumerate(pos_vocab_list):
            pos, _ = tuplee
            vocabulary_pos2index[pos] = i+2
            vocabulary_index2pos[i+2] = pos

        for i,tuplee in enumerate(cap_vocab_list):
            cap, _ = tuplee
            vocabulary_cap2index[cap] = i+1
            vocabulary_index2cap[i+1] = cap
        #print (vocabulary_cap2index)
        #exit()

        for i,tuplee in enumerate(role_1st_label_list):
            label, _ =tuplee
            label = str(label)
            vocabulary_role_1st_label2index[label] = i
            vocabulary_role_1st_index2label[i] = label

        for i,tuplee in enumerate(role_2nd_label_list):
            label, _ =tuplee
            label = str(label)
            vocabulary_role_2nd_label2index[label] = i
            vocabulary_role_2nd_index2label[i] = label

        for i,tuplee in enumerate(func_label_list):
            label, _ =tuplee
            label = str(label)
            vocabulary_func_label2index[label] = i
            vocabulary_func_index2label[i] = label

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,
                             vocabulary_role_1st_label2index,vocabulary_role_1st_index2label,
                             vocabulary_role_2nd_label2index, vocabulary_role_2nd_index2label,
                             vocabulary_func_label2index, vocabulary_func_index2label,
                             vocabulary_char2index, vocabulary_index2char,
                             vocabulary_pos2index, vocabulary_index2pos,
                             vocabulary_cap2index, vocabulary_index2cap), data_f)

    return vocabulary_word2index,vocabulary_index2word, \
           vocabulary_role_1st_label2index, vocabulary_role_1st_index2label,\
           vocabulary_role_2nd_label2index, vocabulary_role_2nd_index2label,\
           vocabulary_func_label2index, vocabulary_func_index2label,\
           vocabulary_char2index, vocabulary_index2char, \
           vocabulary_pos2index, vocabulary_index2pos,\
           vocabulary_cap2index, vocabulary_index2cap


def load_pred_data(data_path,
              vocab_word2index, vocab_char2index, vocab_pos2index, vocab_cap2index,
              sentence_len, word_len,
              flag_use_char, flag_use_pos, flag_use_cap):
    """
    :param data_path:
    :param vocab_label2index:
    :param vocab_word2index:
    :param vocab_char2index:
    :param vocab_pos2index:
    :param vocab_cap2index:
    :param sentence_len: max length of word sequence
    :param word_len:  max length of char sequence
    :return: X: [ word_sequence, char_sequence, pos_sequence, cap_sequence]
                - word_sequence: sentence_len
                - char_sequence: sentence_len * word_len
                - pos_sequence: sentence_len
                - cap_sequence: sentence_len
    """
    data_file = codecs.open(data_path, mode='r', encoding='utf-8')
    data_lines = data_file.readlines()
    # build data samples:
    Word_sequences = []
    Char_sequences = []
    Pos_sequences = []
    Cap_sequences = []
    for i, line in enumerate(data_lines):
        raw_list = line.strip().split("\t")
        input_list = raw_list[1].split(" ")
        # get word lists
        word_sequence = [vocab_word2index.get(x, UNK_ID) for x in input_list]
        Word_sequences.append(word_sequence)
        # get char lists
        if flag_use_char:
            char_sequence = [] # [sentence_len, word_len]
            for word in input_list:
                char_indexs = [vocab_char2index.get(char, UNK_ID) for char in word]
                char_sequence.append(char_indexs)
            if len(input_list) < sentence_len:
                char_sequence.extend( [[0]] * (sentence_len-len(input_list)))
            else:
                char_sequence = char_sequence[:sentence_len]
            char_sequence = pad_sequences(char_sequence, maxlen=word_len, value=0.)
            Char_sequences.append(char_sequence)
        if flag_use_pos:
            pos_sequence = nltk.pos_tag(input_list) # [sentence_len]
            word_seq, pos_seq = zip(*pos_sequence)
            pos_sequence = list(pos_seq)
            pos_sequence = [vocab_pos2index.get(pos, UNK_ID) for pos in pos_sequence]
            Pos_sequences.append(pos_sequence)
        if flag_use_cap:
            cap_sequence = [word_capitalize(word) for word in input_list]
            cap_sequence = [vocab_cap2index[cap] for cap in cap_sequence]
            Cap_sequences.append(cap_sequence)
    Word_sequences = pad_sequences(Word_sequences, maxlen=sentence_len, value=0.)
    if flag_use_pos:
        Pos_sequences = pad_sequences(Pos_sequences, maxlen=sentence_len, value=0.)
    if flag_use_cap:
        Cap_sequences = pad_sequences(Cap_sequences, maxlen=sentence_len, value=0.)
    X = {'word':np.array(Word_sequences), 'char':np.array(Char_sequences), 'pos':np.array(Pos_sequences), 'cap':np.array(Cap_sequences)}
    return X, data_lines






















































