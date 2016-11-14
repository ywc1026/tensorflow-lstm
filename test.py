from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils2 import DataLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    # parser.add_argument('-n', type=int, default=500,
    #                    help='number of characters to sample')
    # parser.add_argument('--prime', type=text_type, default=u' ',
    #                    help='prime text')
    # parser.add_argument('--sample', type=int, default=1,
    #                    help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='data directory containing test.csv')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')

    args = parser.parse_args()
    sample(args)

def sample(args):
    data_loader = DataLoader(args.data_dir, args.batch_size, args.seq_length)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    # with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
    #     chars, vocab = cPickle.load(f)
    model = Model(saved_args)
    total = 0.0
    iters = 0
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                test_loss, state, _, accuracy = sess.run([model.cost, model.final_state, model.train_op, model.accuracy], feed)
                total += accuracy
                iters += 1
                end = time.time()
                print("{}/{} (epoch {}), test_loss = {:.3f}, time/batch = {:.3f}, {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, test_loss, end - start, accuracy))

        print("Accuray is %.3f" % (total / iters))
        

if __name__ == '__main__':
    main()