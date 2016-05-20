import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.framework import dtypes

from model_rnn import ElmanRNN


class BiRNN (ElmanRNN):

    
        
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
        cell_fw = cell_bw = lstm_cell

        initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
          embedding = tf.get_variable("embedding", [self.vocab_size, size])
          inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        
        if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)
        

        # Build RNN
        inputs = [tf.squeeze(input_, [1])   for input_ in tf.split(1, n_steps, inputs)]
        #outputs, state = rnn.rnn(cell, inputs, initial_state=initial_state)
        outputs_pair, state_fw, state_bw = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs, \
                      initial_state_fw=initial_state, initial_state_bw=initial_state)
        #              dtype=None, sequence_length=None, scope=None):

        outputs = []
        for out in outputs_pair:
            out_fw, out_bw = tf.split(1, 2, out)
            outputs.append (1.0 * out_fw + 0.0 * out_bw)
        #outputs_fw = [ [0] for out in outputs_pair]
        self.outputs = outputs
        return self.outputs