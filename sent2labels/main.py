import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

from tensorflow.python.framework import dtypes
from easydict import EasyDict as edict

import data_utils

'''

num_encoder_symbols = 
embedding_size = 
dtype = dtypes.float32

with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq"):
# Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)


'''

# max length of sentences
num_steps = n_steps = seq_length = 5
# num of sentences (one word per sentence) processed in each iteration 
batch_size = 1

# number of symbols used in sentences
vocab_size = 7
#encoding the symbols into 50 dim vectors
embedding_dim = 50

# rnn hidden memory size
n_hidden = memory_dim = 100

#number of inputs = length of input vector
n_input = 1

# num of putput labels
num_tag_labels = 10



_inputs = tf.placeholder(tf.int32, [batch_size, n_steps])
_targets = tf.placeholder(tf.int32, [batch_size, n_steps])

def build_model (inputs, config, is_training=True):
    size = n_hidden

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0)

    # add dropout to output
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    
    #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    cell = lstm_cell

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, inputs)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)


    # Build RNN
    inputs = [tf.squeeze(input_, [1])   for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = rnn.rnn(cell, inputs, initial_state=initial_state)

    return outputs, state


def build_loss_optimizer (outputs, targets, size, is_training = True):
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, num_tag_labels])
    softmax_b = tf.get_variable("softmax_b", [num_tag_labels])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(_targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss) / batch_size

    if not is_training:
      return

    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    # train_op = optimizer.minimize(cost)

    return train_op, loss

def train (train_op, loss, X, Y, config):
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        #saver = tf.train.Saver(tf.all_variables())

        for e in range(config.num_epochs):
            #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            batch_x, batch_y = data_utils.get_next_batch (X, Y, batch_size)
            feed_dict = {
                x: batch_x, y: batch_y
            }

            _, loss,  = sess.run ([train_op, loss], feed_dict)

            print ('epoch ', e)
            print ('loss ', loss)


config = edict({'keep_prob': 0.5, 'num_layers': 1, 'max_grad_norm': 5, 'num_epochs': 100})


outputs, state = build_model (_inputs, config, is_training = True)

train_op, loss = build_loss_optimizer (outputs, _targets, n_hidden)

X, Y = data_utils.get_all_data()

train(train_op, loss, X, Y, config)






'''

# tf Graph input
x = tf.placeholder("int32", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_classes])

# Define weights for hidden and output layers
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])) #output layer weights
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN2(_X, _istate, _weights, _biases):

    # input _X shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

'''

'''
inputs = tf.placeholder(tf.int32, [None, n_steps])
X = tf.transpose(inputs, [1, 0]) #permute  batch_size and n_steps
X = tf.split(0, n_steps, inputs) #n_steps * (batch_size, n_hidden)

# OR tf.split(1, n_steps, inputs)

cell = rnn_cell.GRUCell (n_hidden)
istate = tf.zeros ( (batch_size, n_hidden) )

outputs, states = rnn.rnn (cell, X, initial_state = istate)

'''