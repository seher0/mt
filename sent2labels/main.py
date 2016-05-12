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

class Sent2Labels ():
    def __init__ (self, config):
        self.num_steps = config.num_steps
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden
        self.num_tag_labels = config.num_tag_labels
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
          embedding = tf.get_variable("embedding", [vocab_size, size])
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
        num_tag_labels = self.num_tag_labels
        size = self.n_hidden
        config = self.config


        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, num_tag_labels])
        softmax_b = tf.get_variable("softmax_b", [num_tag_labels])
        logits = tf.matmul(output, softmax_w) + softmax_b
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

    def train (self, sess):
       
        #saver = tf.train.Saver(tf.all_variables())

        for m in range(5):

            e = 0
            data_gen, _, _ = data_utils.get_all_data()
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


config = edict({'keep_prob': 0.5, 'num_layers': 1, 'max_grad_norm': 5, 'num_epochs': 100,
    'num_steps': -1, 
    'vocab_size': -1, 
    'batch_size': 1, 
    'embedding_dim': 50, 
    'n_hidden': 100, 
    'num_tag_labels': 10

    })

gen, max_sent_len, vocab_size = data_utils.get_all_data()
config.num_steps = max_sent_len
config.vocab_size = vocab_size


with tf.Graph().as_default():
    model = Sent2Labels (config)

    _, _ = model.build_model (is_training = True)
    
    train_op, loss = model.build_loss_optimizer (model.outputs)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        model.train(session)






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