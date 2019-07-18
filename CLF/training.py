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
import os
sys.path.insert(0,os.getcwd())
from data_utlis import create_vocabulary,load_data
from tflearn.data_utils import pad_sequences

import random,gensim
import pickle
from model import Model
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
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("word_len",20,"max word length")
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
    if FLAGS.use_pre_word_embedding:
        print("using pre-trained word embedding.started.word2vec_model_path:",FLAGS.word2vec_model_path)
        word2vec_model=gensim.models.KeyedVectors.load_word2vec_format(FLAGS.word2vec_model_path)
    else:
        print("Not using pre-trained word embedding")
        word2vec_model=None

    #1.get vocabulary of X and Label
    print("====="*15)
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

    word_vocab_size = len(vocabulary_word2index)
    char_vocab_size = len(vocabulary_char2index)
    pos_vocab_size = len(vocabulary_pos2index)
    cap_vocab_size = len(vocabulary_cap2index)
    role_1st_label_size = len(vocabulary_role_1st_label2index)
    role_2nd_label_size = len(vocabulary_role_2nd_label2index)
    func_label_size = len(vocabulary_func_label2index)
    print("word_vocab_size: ",word_vocab_size)
    print("char_vocab_size: ",char_vocab_size)
    print("pos_vocab_size: ",pos_vocab_size)
    print("cap_vocab_size: ",cap_vocab_size)
    print("role 1st label size: " ,role_1st_label_size)
    print("role 2nd label size: " ,role_2nd_label_size)
    print("func label size: " ,func_label_size)
    role_1st_num_classes = role_1st_label_size
    role_2nd_num_classes = role_2nd_label_size
    func_num_classes = func_label_size

    train_path ={
        "best_role_1st" : FLAGS.role_1st_train_path,
        "best_role_2nd" : FLAGS.role_2nd_train_path,
        "best_func" : FLAGS.func_train_path
    }

    print("========="*15)
    print("Start load data from file...")
    train = load_data(train_path[FLAGS.framework_type],
                      vocabulary_role_1st_label2index, vocabulary_role_2nd_label2index, vocabulary_func_label2index,
                      vocabulary_word2index, vocabulary_char2index, vocabulary_pos2index, vocabulary_cap2index,
                      FLAGS.sentence_len, FLAGS.word_len,
                      FLAGS.use_char_embedding, FLAGS.use_feature_pos, FLAGS.use_feature_cap)
    dev = load_data(FLAGS.dev_path,
                    vocabulary_role_1st_label2index, vocabulary_role_2nd_label2index, vocabulary_func_label2index,
                    vocabulary_word2index, vocabulary_char2index, vocabulary_pos2index, vocabulary_cap2index,
                      FLAGS.sentence_len, FLAGS.word_len,
                      FLAGS.use_char_embedding, FLAGS.use_feature_pos, FLAGS.use_feature_cap)
    test = load_data(FLAGS.test_path,
                      vocabulary_role_1st_label2index, vocabulary_role_2nd_label2index, vocabulary_func_label2index,
                     vocabulary_word2index, vocabulary_char2index, vocabulary_pos2index,vocabulary_cap2index,
                      FLAGS.sentence_len, FLAGS.word_len,
                      FLAGS.use_char_embedding, FLAGS.use_feature_pos, FLAGS.use_feature_cap)

    trainX, train_role_1st_Y, train_role_2nd_Y, train_func_Y = train[0],train[0]['role_1st'],train[0]['role_2nd'],train[0]['func']
    devX, dev_role_1st_Y, dev_role_2nd_Y, dev_func_Y = dev[0],dev[0]['role_1st'],dev[0]['role_2nd'],dev[0]['func']
    testX, test_role_1st_Y, test_role_2nd_Y, test_func_Y = test[0],test[0]['role_1st'],test[0]['role_2nd'],test[0]['func']



    print("Framework type: ", FLAGS.framework_type)
    print("Training set: ", len(train_role_1st_Y))
    print("Developing set: ", len(dev_role_1st_Y))
    print("Testing set: ", len(test_role_1st_Y))
    print("End load data from file!")

    #2.create session.
    print("====="*15)
    print("start create session... ")
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        print("Start initialize multi-task model...")
        print("Task weights: role_1st：%.2f, role_2nd：%.2f, func：%.2f" % (FLAGS.role_1st_weight, FLAGS.role_2nd_weight, FLAGS.func_weight))
        model = Model(role_1st_num_classes, role_2nd_num_classes, func_num_classes,
                      FLAGS.role_1st_weight, FLAGS.role_2nd_weight, FLAGS.func_weight,
                      FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                      FLAGS.sentence_len, word_vocab_size, FLAGS.word_embed_size,
                      flag_use_char_embedding=True, char_vocab_size=char_vocab_size,
                      char_embed_size=FLAGS.char_embed_size, word_len=FLAGS.word_len,
                      flag_use_pos_feature=True, pos_vocab_size=pos_vocab_size, pos_embed_size=FLAGS.feature_embed_size,
                      flag_use_cap_feature=True, cap_vocab_size=cap_vocab_size, cap_embed_size=FLAGS.feature_embed_size,
                      is_training=FLAGS.is_training
                      )
        print("End initialize multi-task model!")
        #exit()
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_pre_word_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, word_vocab_size, model, word2vec_model)
        curr_epoch=sess.run(model.epoch_step)

        #3.feed data & training
        print("Start training!")
        trainX_word, trainX_char, trainX_pos, trainX_cap = trainX['word'], trainX['char'], trainX['pos'], trainX['cap']
        number_of_training_data=len(trainX_word)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            all_loss, loss, acc, counter = 0.0, 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                feed_dict = {model.input_words: trainX_word[start:end],
                             model.input_role_1st_label: train_role_1st_Y[start:end],
                             model.input_role_2nd_label : train_role_2nd_Y[start:end],
                             model.input_func_label : train_func_Y[start:end],
                             model.dropout_keep_prob:0.5}
                if FLAGS.use_char_embedding:
                    feed_dict[model.input_chars] = trainX_char[start:end]
                    #print(trainX_char[start:end].shape)
                if FLAGS.use_feature_pos:
                    feed_dict[model.input_pos] = trainX_pos[start:end]
                if FLAGS.use_feature_cap:
                    feed_dict[model.input_cap] = trainX_cap[start:end]

                # for each framework type
                if FLAGS.framework_type == 'best_role_1st':
                    curr_all_loss, curr_loss, curr_acc, _ =sess.run([model.loss_val,
                                                                     model.role_1st_losses,
                                                                     model.role_1st_accuracy,
                                                                     model.train_op],feed_dict)
                    all_loss, loss, counter, acc = all_loss+curr_all_loss, loss+curr_loss, counter+1, acc+curr_acc
                    print("Epoch %d\tBatch %d\tAll loss:%.3f\tRole 1st loss:%.3f\tRole 1st acc:%.3f" %
                          (epoch, counter, all_loss / float(counter), loss / float(counter), acc / float(counter)))
                if FLAGS.framework_type == 'best_role_2nd':
                    curr_all_loss, curr_loss, curr_acc, _ =sess.run([model.loss_val,
                                                                     model.role_2nd_losses,
                                                                     model.role_2nd_accuracy,
                                                                     model.train_op],feed_dict)
                    all_loss, loss, counter, acc = all_loss+curr_all_loss, loss+curr_loss, counter+1, acc+curr_acc
                    print("Epoch %d\tBatch %d\tAll loss:%.3f\tRole 2nd loss:%.3f\tRole 2nd acc:%.3f" %
                          (epoch, counter, all_loss / float(counter), loss / float(counter), acc / float(counter)))
                if FLAGS.framework_type == 'best_func':
                    curr_all_loss, curr_loss, curr_acc, _ =sess.run([model.loss_val,
                                                                     model.func_losses,
                                                                     model.func_accuracy,
                                                                     model.train_op],feed_dict)
                    all_loss, loss, counter, acc = all_loss+curr_all_loss, loss+curr_loss, counter+1, acc+curr_acc
                    print("Epoch %d\tBatch %d\tAll loss:%.3f\tFunc loss:%.3f\tFunc acc:%.3f" %
                          (epoch, counter, all_loss / float(counter), loss / float(counter), acc / float(counter)))

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0 and epoch!=0:
                print("----------------Epoch %d Validation----------------" % epoch)
                all_loss, role_1st_loss, role_1st_acc, role_1st_f1_macro, role_1st_f1_micro, \
                role_2nd_loss, role_2nd_acc, role_2nd_f1_macro, role_2nd_f1_micro, \
                func_loss, func_acc, func_f1_macro, func_f1_micro = do_eval(sess, model,
                                                                            devX, dev_role_1st_Y, dev_role_2nd_Y, dev_func_Y,
                                                                            role_1st_num_classes, role_2nd_num_classes, func_num_classes)
                print("All loss: %.3f" % all_loss)
                print(">>>>>Role 1st<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (role_1st_loss, role_1st_acc, role_1st_f1_micro, role_1st_f1_macro))
                print(">>>>>Role 2nd<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (role_2nd_loss, role_2nd_acc, role_2nd_f1_micro, role_2nd_f1_macro))
                print(">>>>>>>Func<<<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (func_loss, func_acc, func_f1_micro, func_f1_macro))

                print("----------------Epoch %d Test----------------" % epoch)
                all_loss, role_1st_loss, role_1st_acc, role_1st_f1_macro, role_1st_f1_micro, \
                role_2nd_loss, role_2nd_acc, role_2nd_f1_macro, role_2nd_f1_micro, \
                func_loss, func_acc, func_f1_macro, func_f1_micro = do_eval(sess, model,
                                                                            testX, test_role_1st_Y, test_role_2nd_Y, test_func_Y,
                                                                            role_1st_num_classes, role_2nd_num_classes, func_num_classes)
                print("All loss: %.3f" % all_loss)
                print(">>>>>Role 1st<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (
                role_1st_loss, role_1st_acc, role_1st_f1_micro, role_1st_f1_macro))
                print(">>>>>Role 2nd<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (
                role_2nd_loss, role_2nd_acc, role_2nd_f1_micro, role_2nd_f1_macro))
                print(">>>>>>>Func<<<<<<<")
                print("Loss:%.3f\tACC:%.3f\tF1-micro:%.3f\tF1-macro:%.3f" % (
                func_loss, func_acc, func_f1_micro, func_f1_macro))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model):
    #import word2vec # we put import here so that many people who do not use word2vec do not need to install this package. you can move import to the beginning of this file.
    print("using pre-trained word emebedding.started")
    word_embedding_2dlist = [[]] * (vocab_size+1)  # create an empty word_embedding list.
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    word_embedding_2dlist[0] = np.zeros(FLAGS.word_embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(FLAGS.word_embed_size)
    word_embedding_2dlist[2] = np.random.uniform(-bound, bound, FLAGS.word_embed_size)
    count_exist = 0
    count_not_exist = 0
    for i in range(3, vocab_size):  # loop each word. notice that the first 3 words are pad and unknown token
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_model[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.word_embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_2dlist[vocab_size] = np.zeros(FLAGS.word_embed_size)
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, eval_role_1st_Y, eval_role_2nd_Y, eval_func_Y,
                role_1st_num_classes, role_2nd_num_classes, func_num_classes):
    evalX_word, evalX_char, evalX_pos, evalX_cap = evalX['word'], evalX['char'], evalX['pos'], evalX['cap']
    #label_dict_confuse_matrix=init_label_dict(num_classes)
    feed_dict = {model.input_words: evalX_word,
                 model.input_role_1st_label: eval_role_1st_Y,
                 model.input_role_2nd_label : eval_role_2nd_Y,
                 model.input_func_label : eval_func_Y,
                 model.dropout_keep_prob: 1}
    if FLAGS.use_char_embedding:
        feed_dict[model.input_chars] = evalX_char
    if FLAGS.use_feature_pos:
        feed_dict[model.input_pos] = evalX_pos
    if FLAGS.use_feature_cap:
        feed_dict[model.input_cap] = evalX_cap
    all_loss,\
    role_1st_acc, role_1st_predict_y, role_1st_loss,\
    role_2nd_acc, role_2nd_predict_y, role_2nd_loss,\
    func_acc, func_predict_y, func_loss  = sess.run([model.loss_val,
                                                    model.role_1st_accuracy, model.role_1st_predictions, model.role_1st_losses,
                                                    model.role_2nd_accuracy, model.role_2nd_predictions, model.role_2nd_losses,
                                                    model.func_accuracy, model.func_predictions, model.func_losses], feed_dict)
    # results for role 1st:
    role_1st_target_y = eval_role_1st_Y
    confuse_matrix=compute_confuse_matrix(role_1st_target_y, role_1st_predict_y, role_1st_num_classes)
    role_1st_f1_micro = compute_micro_f1(confuse_matrix)
    role_1st_f1_macro = compute_macro_f1(confuse_matrix)
    # results for role 2nd:
    role_2nd_target_y = eval_role_2nd_Y
    confuse_matrix=compute_confuse_matrix(role_2nd_target_y, role_2nd_predict_y, role_2nd_num_classes)
    role_2nd_f1_micro = compute_micro_f1(confuse_matrix)
    role_2nd_f1_macro = compute_macro_f1(confuse_matrix)
    # results for func:
    func_target_y = eval_func_Y
    confuse_matrix=compute_confuse_matrix(func_target_y, func_predict_y, func_num_classes)
    func_f1_micro = compute_micro_f1(confuse_matrix)
    func_f1_macro = compute_macro_f1(confuse_matrix)
    return all_loss, role_1st_loss, role_1st_acc, role_1st_f1_macro, role_1st_f1_micro, \
                    role_2nd_loss, role_2nd_acc, role_2nd_f1_macro, role_2nd_f1_micro,\
                    func_loss, func_acc, func_f1_macro, func_f1_micro

#######################################
def compute_confuse_matrix(predict_y, target_y, num_classes):
    labels = dict()
    for label in range(num_classes):
        labels[label] = {'TP':0, 'FP':0, 'FN':0}
    for pred, tar in zip(predict_y, target_y):
        if pred == tar:
            labels[pred]['TP'] += 1
        else:
            labels[pred]['FP'] += 1
            labels[tar]['FN'] += 1
    return labels

def compute_precision_recall(TP, FP, FN):
    small_value = 0.000001
    precision=TP/(TP+FP+small_value)
    recall=TP/(TP+FN+small_value)
    return precision, recall

def compute_micro_f1(confuse_matrix):
    all_TP = 0
    all_FP = 0
    all_FN = 0
    for label in confuse_matrix:
        all_TP += confuse_matrix[label]['TP']
        all_FP += confuse_matrix[label]['FP']
        all_FN += confuse_matrix[label]['FN']
    precision, recall = compute_precision_recall(all_TP/len(confuse_matrix), all_FP/len(confuse_matrix), all_FN/len(confuse_matrix))
    return compure_f1(precision, recall)

def compute_macro_f1(confuse_matrix):
    all_f1 = 0
    for label in confuse_matrix:
        TP = confuse_matrix[label]['TP']
        FP = confuse_matrix[label]['FP']
        FN = confuse_matrix[label]['FN']
        precision, recall = compute_precision_recall(TP, FP, FN)
        f1 = compure_f1(precision, recall)
        #print(label, f1)
        all_f1 += f1
    return all_f1 / len(confuse_matrix)

def compure_f1(precision, recall):
    small_value = 0.000001
    return precision*recall*2 / (precision + recall + small_value)

if __name__ == "__main__":
    tf.app.run()






































