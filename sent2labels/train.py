import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from easydict import EasyDict as edict

import data_utils
import json

'''


with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq"):
# Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)


'''
'''

# max length of sentences
num_steps = n_steps = seq_length = 5
# num of sentences (one word per sentence) processed in each iteration 
batch_size = 1

# number of symbols used in sentences
#vocab_size = 7
#encoding the symbols into 50 dim vectors
embedding_dim = 50

# rnn hidden memory size
n_hidden = memory_dim = 100

#number of inputs = length of input vector
n_input = 1

# num of putput labels
num_tag_labels = 10

'''

from model_rnn import ElmanRNN
from model_birnn import BiRNN


config = edict({'keep_prob': 0.5, 'num_layers': 1, 'max_grad_norm': 5, 'num_epochs': 100,
    'num_steps': -1, 
    'vocab_size': -1, 
    'batch_size': 1, 
    'embedding_dim': 50, 
    'n_hidden': 100, 
    'num_tag_labels': 10

    })

gen, max_sent_len, vocab, label_vocab = data_utils.get_all_data()
config.num_steps = max_sent_len
config.vocab_size = len(vocab)
config.label_vocab_size = len(label_vocab)


with tf.Graph().as_default():
    model = ElmanRNN (config)
    #model = BiRNN (config)

    _ = model.build_model (is_training = True)
    
    train_op, loss = model.build_loss_optimizer (model.outputs)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        model.train(session, iterations = 20)

    with open('models/config.json', 'wb') as f:
        json.dump(config, f, indent=4 )


    with open('models/vocab.json', 'wb') as f:
        json.dump(vocab, f, indent=4 )

    with open('models/label_vocab.json', 'wb') as f:
        json.dump(label_vocab, f, indent=4 )





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