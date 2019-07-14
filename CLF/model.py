# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Model(object):
    def __init__(self,role_1st_num_classes, role_2nd_num_classes, func_num_classes,
                 role_1st_weight, role_2nd_weight, func_weight,
                 learning_rate, batch_size, decay_steps, decay_rate,
                 sentence_len, word_vocab_size, word_embed_size,
                 flag_use_char_embedding=False, char_vocab_size=None, char_embed_size=None, word_len = None,
                 flag_use_pos_feature=False, pos_vocab_size=None, pos_embed_size=None,
                 flag_use_cap_feature=False, cap_vocab_size=None, cap_embed_size=None,
                 is_training=True,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparameter
        self.role_1st_num_classes = role_1st_num_classes
        self.role_2nd_num_classes = role_2nd_num_classes
        self.func_num_classes = func_num_classes

        self.role_1st_weight =  role_1st_weight
        self.role_2nd_weight = role_2nd_weight
        self.func_weight = func_weight

        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.word_vocab_size = word_vocab_size
        self.word_embed_size = word_embed_size

        self.use_char_embedding = flag_use_char_embedding
        self.use_pos_feature = flag_use_pos_feature
        self.use_cap_feature = flag_use_cap_feature
        if self.use_char_embedding:
            self.char_vocab_size = char_vocab_size
            self.char_embed_size = char_embed_size
            self.word_len = word_len
        else:
            self.char_vocab_size = None
            self.char_embed_size = None
            self.word_len = None
        if self.use_pos_feature:
            self.pos_vocab_size = pos_vocab_size
            self.pos_embed_size = pos_embed_size
        else:
            self.pos_vocab_size = None
            self.pos_embed_size = None
        if self.use_cap_feature:
            self.cap_vocab_size = cap_vocab_size
            self.cap_embed_size = cap_embed_size
        else:
            self.cap_vocab_size = None
            self.cap_embed_size = None

        self.hidden_size = 50
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.num_sampled = 20

        self.input_words = tf.placeholder(tf.int32, [None, self.sentence_len], name="input_words")  # word sequences
        self.input_role_1st_label = tf.placeholder(tf.int32, [None, ], name="input_role_1st_label")  # labels [None,num_classes]
        self.input_role_2nd_label = tf.placeholder(tf.int32, [None, ], name="input_role_2nd_label")
        self.input_func_label = tf.placeholder(tf.int32, [None, ], name="input_func_label")
        self.input_chars = tf.placeholder(tf.int32, [None, self.sentence_len, self.word_len], name="input_chars")
        self.input_pos = tf.placeholder(tf.int32, [None, self.sentence_len], name="input_pos")
        self.input_cap = tf.placeholder(tf.int32, [None, self.sentence_len], name="input_cap")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.role_1st_logits, self.role_2nd_logits, self.func_logits= self.inference() #[None, self.label_size]. main computation graph is here.
        if is_training:
            self.loss_val = self.loss()   # weighted loss
            self.train_op = self.train()
        # metrics for role 1st
        self.role_1st_predictions = tf.argmax(self.role_1st_logits, axis=1, name="role_1st_predictions")  # shape:[None,]
        role_1st_correct_prediction = tf.equal(tf.cast(self.role_1st_predictions, tf.int32), self.input_role_1st_label) #tf.argmax(self.logits, 1)-->[batch_size]
        self.role_1st_accuracy = tf.reduce_mean(tf.cast(role_1st_correct_prediction, tf.float32), name="role_1st_accuracy") # shape=()
        # metrics for role 2nd
        self.role_2nd_predictions = tf.argmax(self.role_2nd_logits, axis=1, name="role_2nd_predictions")  # shape:[None,]
        role_2nd_correct_prediction = tf.equal(tf.cast(self.role_2nd_predictions, tf.int32), self.input_role_2nd_label) #tf.argmax(self.logits, 1)-->[batch_size]
        self.role_2nd_accuracy = tf.reduce_mean(tf.cast(role_2nd_correct_prediction, tf.float32), name="role_2nd_accuracy") # shape=()
        # metrics for role 1st
        self.func_predictions = tf.argmax(self.func_logits, axis=1, name="func_predictions")  # shape:[None,]
        func_correct_prediction = tf.equal(tf.cast(self.func_predictions, tf.int32), self.input_func_label) #tf.argmax(self.logits, 1)-->[batch_size]
        self.func_accuracy = tf.reduce_mean(tf.cast(func_correct_prediction, tf.float32), name="func_accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("word_embedding"): # embedding matrix
            self.Embedding = tf.get_variable("WordEmbedding",shape=[self.word_vocab_size+1, self.word_embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)


        if self.use_char_embedding:
            with tf.name_scope("char_embedding"):
                self.CharEmbedding = tf.get_variable("CharEmbedding", shape=[self.char_vocab_size+1, self.char_embed_size], initializer=self.initializer)
        if self.use_pos_feature:
            with tf.name_scope("pos_embedding"):
                self.PosEmbedding = tf.get_variable("PosEmbedding", shape=[self.pos_vocab_size+1, self.pos_embed_size], initializer=self.initializer)
        if self.use_cap_feature:
            with tf.name_scope("cap_embedding"):
                self.CapEmbedding = tf.get_variable("CapEmbedding", shape=[self.pos_vocab_size+1, self.pos_embed_size], initializer=self.initializer)


        with tf.name_scope("attention_linear"):
            self.w_attention_1 = tf.get_variable("w_attention_1", shape=[self.hidden_size * 2, 1])
            self.b_attention_1 = tf.get_variable("b_attention_1", shape=[1])
            #self.w_attention_2 = tf.get_variable("w_attention_2", shape=[self.hidden_size, 1])
            #self.b_attention_2 = tf.get_variable("b_attention_2", shape=[1])

        # with tf.name_scope("role_1st_attention_linear"):
        #     self.w_role_1st_attention_1 = tf.get_variable("w_role_1st_attention_1", shape=[self.hidden_size * 2, 1])
        #     self.b_role_1st_attention_1 = tf.get_variable("b_role_1st_attention_1", shape=[1])
        #     #self.w_attention_2 = tf.get_variable("w_attention_2", shape=[self.hidden_size, 1])
        #     #self.b_attention_2 = tf.get_variable("b_attention_2", shape=[1])
        # with tf.name_scope("role_2nd_attention_linear"):
        #     self.w_role_2nd_attention_1 = tf.get_variable("w_role_2nd_attention_1", shape=[self.hidden_size * 2, 1])
        #     self.b_role_2nd_attention_1 = tf.get_variable("b_role_2nd_attention_1", shape=[1])
        #     #self.w_attention_2 = tf.get_variable("w_attention_2", shape=[self.hidden_size, 1])
        #     #self.b_attention_2 = tf.get_variable("b_attention_2", shape=[1])
        # with tf.name_scope("func_attention_linear"):
        #     self.w_func_attention_1 = tf.get_variable("w_func_attention_1", shape=[self.hidden_size * 2, 1])
        #     self.b_func_attention_1 = tf.get_variable("b_func_attention_1", shape=[1])
        #     #self.w_attention_2 = tf.get_variable("w_attention_2", shape=[self.hidden_size, 1])
        #     #self.b_attention_2 = tf.get_variable("b_attention_2", shape=[1])

        with tf.name_scope("role_1st_output_linear"):
            self.W_role_1st_projection = tf.get_variable("W_role_1st_projection", shape=[self.hidden_size * 2, self.role_1st_num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_role_1st_projection = tf.get_variable("b_role_1st_projection", shape=[self.role_1st_num_classes])  # [label_size]

        with tf.name_scope("role_2nd_output_linear"):
            self.W_role_2nd_projection = tf.get_variable("W_role_2nd_projection", shape=[self.hidden_size * 2, self.role_2nd_num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_role_2nd_projection = tf.get_variable("b_role_2nd_projection", shape=[self.role_2nd_num_classes])  # [label_size]

        with tf.name_scope("func_output_linear"):
            self.W_func_projection = tf.get_variable("W_func_projection", shape=[self.hidden_size * 2, self.func_num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_func_projection = tf.get_variable("b_func_projection", shape=[self.func_num_classes])  # [label_size]


    def inference(self):
        """main computation graph here: 1. char presentation layer, 2. word presentation layer 3. BiLSTM layer, 4. word attention layer 5.softmax layer"""
        #0. get embeddings of words, chars and features in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_words) #shape:[None, sentence_len, embed_size]
        if self.use_char_embedding:
            self.embedded_chars = tf.nn.embedding_lookup(self.CharEmbedding, self.input_chars) #shape:[None, sentence_len, word_len, embed_size]
        if self.use_pos_feature:
            self.embedded_pos = tf.nn.embedding_lookup(self.PosEmbedding, self.input_pos) #shape: [None, sentence_len, embed_size]
        if self.use_cap_feature:
            self.embedded_cap = tf.nn.embedding_lookup(self.CapEmbedding, self.input_cap) #shape: [None, sentence_len, embed_size]

        #2. word presentation layer
        embedding_list = [self.embedded_words]
        # 1. char presentation layer
        if self.use_char_embedding:
            char_presentation = self.char_BiLSTM()  # [batch_size, sentence_len, hidden_size*2]
            embedding_list.append(char_presentation)
        if self.use_pos_feature:
            embedding_list.append(self.embedded_pos)
        if self.use_cap_feature:
            embedding_list.append(self.embedded_cap)
        word_presentation = tf.concat(embedding_list, axis=2)
        # [batch_size, sentence_len, word_embed_size + hidden_size*2 + pos_embed_size + cap_embed_size]
        #3. BiLSTM layer
        lstm_output = self.word_BiLSTM(word_presentation) # [batch_size, sentence_len, hidden_size*2]
        # #4. attention layers
        # role_1st_attention_output = self.word_attention(lstm_output, self.w_role_1st_attention_1, self.b_role_1st_attention_1)
        # role_2nd_attention_output = self.word_attention(lstm_output, self.w_role_2nd_attention_1, self.b_role_2nd_attention_1)
        # func_attention_output = self.word_attention(lstm_output, self.w_func_attention_1, self.b_func_attention_1)
        # #5. output layer
        # with tf.name_scope("role_1st_output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        #     role_1st_logits = tf.matmul(role_1st_attention_output, self.W_role_1st_projection) + self.b_role_1st_projection  # [batch_size,role_1st_num_classes]
        # with tf.name_scope("role_2nd_output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        #     role_2nd_logits = tf.matmul(role_2nd_attention_output, self.W_role_2nd_projection) + self.b_role_2nd_projection  # [batch_size,role_2nd_num_classes]
        # with tf.name_scope("func_output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        #     func_logits = tf.matmul(func_attention_output, self.W_func_projection) + self.b_func_projection  # [batch_size,role_1st_num_classes]

        attention_output = self.word_attention(lstm_output, self.w_attention_1, self.b_attention_1)
        with tf.name_scope("role_1st_output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            role_1st_logits = tf.matmul(attention_output, self.W_role_1st_projection) + self.b_role_1st_projection  # [batch_size,role_1st_num_classes]
        with tf.name_scope("role_2nd_output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            role_2nd_logits = tf.matmul(attention_output,self.W_role_2nd_projection) + self.b_role_2nd_projection  # [batch_size,role_2nd_num_classes]
        with tf.name_scope("func_output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            func_logits = tf.matmul(attention_output,self.W_func_projection) + self.b_func_projection  # [batch_size,role_1st_num_classes]

        return role_1st_logits, role_2nd_logits, func_logits

    def char_BiLSTM(self):
        with tf.variable_scope("char_bilstm"):
            embedded_chars = tf.reshape(self.embedded_chars, shape=[-1, self.word_len, self.char_embed_size]) # [batch_size * sentence_len, word_len, embed_size]
            # BiLSTM operation
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                         embedded_chars,
                                                        dtype=tf.float32)
            # [batch_size * sentence_len, word_len, hidden_size]
            # #creates a dynamic bidirectional recurrent neural network
            output_rnn = tf.concat(outputs, axis=2) # [batch_size * sentence_len, word_len, hidden_size*2]
            output_rnn_last = output_rnn[:,-1,:] # [batch_size * sentence_len, hidden_size*2]
            output_rnn_last = tf.reshape(output_rnn_last, shape=[-1, self.sentence_len, self.hidden_size*2]) # [batch_size, sentence_len, hidden_size*2]
            return output_rnn_last

    def word_BiLSTM(self, word_presentation):
        with tf.variable_scope("word_bilstm"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                         word_presentation,
                                                         dtype=tf.float32)
            # [batch_size , sentence_len, hidden_size]
            # #creates a dynamic bidirectional recurrent neural network
            output_rnn = tf.concat(outputs, axis=2)  # [batch_size, sentence_len, hidden_size*2]
            #output_rnn_last = output_rnn[:, -1, :]  # [batch_size, hidden_size*2]
            return output_rnn

    def word_attention(self, hidden_state, w_attention, b_attention):
        """
            input: word_presentation [batch_size, sentence_len, hidden_size*2]
           :return: representation [batch_size, hidden_size*2]
        """
        # 0) one layer of feed forward network
        hidden_state_ = tf.reshape(hidden_state, shape=[-1, self.hidden_size * 2]) # [batch_size*sentence_len, hidden_size*2]

        hidden_state_1 = tf.nn.tanh(tf.matmul(hidden_state_, w_attention) + b_attention)
        #hidden_state_2 = tf.nn.tanh(tf.matmul(hidden_state_1, self.w_attention_2) + self.b_attention_2)
        # [batch_size*sentence_len, 1]
        hidden_logits = tf.reshape(hidden_state_1, shape=[-1, self.sentence_len,1])
        # [batch_size, sentence_len, 1]

        # attention process:
        # 1.get logits for each word in the sentence.
        # 2.get possibility distribution for each word in the sentence.
        # 3.get weighted sum for the sentence as sentence representation.
        # 1) get logits for each word in the sentence.
        #attention_logits = tf.reduce_sum(hidden_representation, axis=2)
        # shape:[batch_size, sentence_len]
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        #attention_logits_max = tf.reduce_max(attention_logits, axis=1,keep_dims=True)  # shape:[batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        p_attention = tf.nn.softmax(hidden_logits)
        # [batch_size, sentence_len, 1]

        # 3) get weighted hidden state by attention vector
        #p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention, hidden_state)
        # [batch_size, sequence_len, hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,axis=1)
        # [batch_size, hidden_size*2]
        return sentence_representation
        # [batch_size, hidden_size*2]

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            self.role_1st_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_role_1st_label, logits=self.role_1st_logits)
            self.role_2nd_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_role_2nd_label, logits=self.role_2nd_logits)
            self.func_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_func_label, logits=self.func_logits)
            losses = self.role_1st_losses * self.role_1st_weight + \
                     self.role_2nd_losses * self.role_2nd_weight + \
                     self.func_losses * self.func_weight
            #sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            self.role_1st_losses = tf.reduce_mean(self.role_1st_losses)
            self.role_2nd_losses = tf.reduce_mean(self.role_2nd_losses)
            self.func_losses = tf.reduce_mean(self.func_losses)
            loss=tf.reduce_mean(losses)     #print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

