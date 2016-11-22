"""Examples for building a stocks LSTM model.

    Imitate the char-rnn-tensorflow
https://github.com/Guinsoon/char-rnn-tensorflow
"""

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class Model():
    """Generate the Model"""
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        # if is_training:
        #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        softmax_w = tf.get_variable("softmax_w", [args.rnn_size, 1])
        softmax_b = tf.get_variable("softmax_b", [1])
 
        iw = tf.get_variable("input_w", [1,args.rnn_size])
        ib = tf.get_variable("input_b", [args.rnn_size])
        inputs = [(tf.matmul(inputs_, iw)+ib) for inputs_ in tf.split(1, args.seq_length, self.input_data)]

        # if is_training:
        #     inputs = [tf.nn.dropout(input_, 0.5) for input_ in inputs]

        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)
        rnn_output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        logits = tf.matmul(rnn_output, softmax_w) + softmax_b
        self.cost = tf.reduce_mean(tf.square(logits - tf.reshape(self.targets, [-1])))
        correct_prediction = tf.less_equal(tf.abs((logits - tf.reshape(self.targets, [-1]))), 
            tf.constant(0.2, shape=[args.batch_size * args.seq_length]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.final_state = state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))








