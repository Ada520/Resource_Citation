# _*_ coding:utf-8 _*_

# triainig the model
'''
process:
1.load data(X: list of lint, y:int)
2.create session
3. feed data
4.training
5.validation
6.prediction
'''
import os
import sys

import tensorflow as tf
import numpy as np

import random,gensim
import os
import pickle
from data_utlis import create_vocabulary#load_data
from tflearn.data_utils import pad_sequences
import os
sys.path.insert(0,os.getcwd())
import random,gensim
import pickle
# configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("framework_type",'best_role_1st','type of learning object')
#可选项为best_1st_role/best_2nd_role/best_func
#-----------------path of datasets--------
tf.app.flags.DEFINE_string("role_1st_train_path",'../data/SciRes/role_1st_train.txt','path of 1st role training data.')
tf.app.flags.DEFINE_string("role_2nd_train_path",'../data/SciRes/role_2nd_train.txt','path of 2nd role training data.')
tf.app.flags.DEFINE_string('func_train_path','../data/SciRes/role_2nd_train.txt','path of func training data.')
tf.app.flags.DEFINE_string("dev_path",'../data/SciRes/dev.txt','path of developing data.')
tf.app.flags.DEFINE_string("test_path",'../data/SciRes/test.txt','path of testing data.')

#-------------path pf pretrained word embedding -----------------------
tf.app.flags.DEFINE_string("word2vec_model_path",'sci.vector',"word2vect's vocabulary and vectors.")
tf.app.flags.DEFINE_string("glove_,odel_path",'glove6B.200d.txt',"Glove's vocabulary and vectors")

#---------------------------path of save models------------------------------------
tf.app.flags.DEFINE_string("ckpt_dir",'./checkpoints/'+FLAGS.framework_type+'/','checkpoint location for the model')
tf.app.flags.DEFINE_string("name_scope",'att_lstm',"name scope value")

#---------------------------parameters----------------------------
tf.app.flags.DEFINE_float('role_1st_weight',0.3,'weight of role 1st classification task')
tf.app.flags.DEFINE_float('role_2nd_weight',0.3,"weight of role 2nd claddification task")
tf.app.flags.DEFINE_float('func_weight',0.3,'weight of func classification task')

tf.app.flags.DEFINE_integer('word_embed_size',200,"word embedding size")
tf.app.flags.DEFINE_integer("char_embed_size",200,"char embedding size")
tf.app.flags.DEFINE_integer("feature_embed_size",50,"feature embedding size")
tf.app.flags.DEFINE_integer("sentence_len",20,"max word length")
tf.app.flags.DEFINE_integer("learning_rate",1e-2,'learning rate')
tf.app.flags.DEFINE_integer("batch_size",512,"Batch size for trainining/evaluating")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("num_epochs",150,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true: training, false:testing/inference")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pre_word_embedding", False, "whether to use embedding or not.")
tf.app.flags.DEFINE_boolean("use_char_embedding", True, "Whether to use char embedding or not.")
tf.app.flags.DEFINE_boolean("use_feature_pos", True, "Whether to use POS feature or not")
tf.app.flags.DEFINE_boolean("use_feature_cap", True, "Whether to use CAP feature or not")

def main(_):
    #0.load pre-training word embeddings
    # move into 1. create_vocabulary
    if FLAGS.use_pre_word_embdding:
        print("using pre-trained word embedding.started.word2vec_model_path:",FLAGS.word2vec_model_path)
        word2vec_model=gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec_model_path)
    else:
        print("Not using pre-trained word embedding")
        word2vec_model=None

    #1.get vocabulary of X and Label
    print("strart create vocabulary...")
    vocabulary_word2index,vocabulary_index2word,\
    vocabulary_role_1st_label2index,vocabulary_role_1st_index2label,\
    vocabulary_role_2nd_label2index,vocabulary_role_2nd_index2label,\
    vocabulary_func_label2index,vocabulary_func_index2label,\
    vocabulary_char2index, vocabulary_index2char,\
    vocabulary_pos2index, vocabulary_index2pos ,\
    vocabulary_cap2index, vocabulary_index2cap = create_vocabulary(FLAGS.role_1st_train_path, word2vec_model)
    print("End create vocabulary!")
    print("Role 1st labels: ", vocabulary_role_1st_index2label)
    print("Role 2nd labels: ", vocabulary_role_2nd_index2label)
    print("Func labels: ", vocabulary_func_index2label)





































