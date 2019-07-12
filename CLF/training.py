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
import sys
sys.path.insert(0,os.getcwd(0))
import tensorflow as tf
import numpy as np
import os
import random,gensim
import os
import pickle
from data_utils import load_data,create_vocabulary
from tflearn.data_utils import pad_sequences
import os
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





































