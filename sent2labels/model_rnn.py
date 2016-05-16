import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.framework import dtypes

import data_utils


class Sent2Labels ():
    def __init__ (self, config):
        self.num_steps = config.num_steps
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden
        self.label_vocab_size = config.label_vocab_size
        self.vocab_size = config.vocab_size
        self.config = config

        self._inputs = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])


    def build_model (self, is_training=True):
        batch_size = self.batch_size
        n_steps = self.num_steps
        size = self.n_hidden
        config = self.config


        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)

        
        # add dropout to output
        if is_training and config.keep_prob < 1:
          lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=config.keep_prob)
        

        #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        cell = lstm_cell

        initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
          embedding = tf.get_variable("embedding", [self.vocab_size, size])
          inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        
        if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)
        

        # Build RNN
        inputs = [tf.squeeze(input_, [1])   for input_ in tf.split(1, n_steps, inputs)]
        outputs, state = rnn.rnn(cell, inputs, initial_state=initial_state)

        self.outputs = outputs
        self.final_state = state

        return outputs, state


    def build_loss_optimizer (self, outputs, is_training = True):

        batch_size = self.batch_size
        num_steps = self.num_steps

        size = self.n_hidden
        config = self.config


        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, self.label_vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.label_vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        self.logits = logits
        self.probs = tf.nn.softmax(self.logits)

        lossf = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        
        self.loss = tf.reduce_sum(lossf) / batch_size



        if not is_training:
          return

        #learning_rate = tf.Variable(float(0.01), trainable=False)
        lr = tf.Variable(0.01, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        # train_op = optimizer.minimize(cost)

        return self.train_op, self.loss

    def train (self, sess, iterations = 5, checkpoint_path='./models/model.ckpt'):
       
        saver = tf.train.Saver(tf.all_variables())

        for m in range(iterations):

            e = 0
            data_gen, _, _, _ = data_utils.get_all_data()
            while True:
                e += 1

                #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                try:
                    batch_x, batch_y = data_utils.get_next_batch (data_gen, self.batch_size)
                    batch_x = np.array([batch_x])
                    batch_y = np.array([batch_y])

                    #print batch_x, batch_y.shape

                except:
                    print 'breaking..'
                    break

                feed_dict = {
                    self._inputs: batch_x, 
                    self._targets: batch_y
                }

                _, loss  = sess.run ([self.train_op, self.loss], feed_dict)

             
                print ('iteration %d.%d, loss %f' % (m,e,loss) )

        saver.save(sess, checkpoint_path)


    def load (self, sess, ckpt_path = './models/model.ckpt'):
        saver = tf.train.Saver(tf.all_variables())

        saver.restore(sess, ckpt_path)


    def test (self, sess, batch_x):

        feed_dict = {
            self._inputs: batch_x
        }

        outputs = sess.run ([self.probs], feed_dict)

        return outputs