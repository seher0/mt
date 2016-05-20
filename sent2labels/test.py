import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from easydict import EasyDict as edict

import data_utils
import json



from model_rnn import ElmanRNN
from model_birnn import BiRNN
from easydict import EasyDict as edict



sents = [
    'I want to recharge',
    'recharge my phone for 100 rs',
    'recharge 9988776655 for rs 100',
    'rs 50',
    'Can I get Rs 200 recharge for 9912312345',
    'pay my postpaid bill',
    'book me a bus from siliguri to delhi',
    'cab from home to work'
]



def init ():
    with open('models/config.json', 'r') as f:
        config = json.load(f)
        config = edict(config)
        print config

    max_sent_len = config['num_steps']

    with open('models/vocab.json', 'r') as f:
        vocab = json.load(f)
        #vocab = edict(vocab)
        print 'vocab -- ', vocab


    with open('models/label_vocab.json', 'r') as f:
        label_vocab = json.load(f)
        #vocab = edict(vocab)
        print 'label_vocab -- ', label_vocab
        inv_label_vocab = {v : k  for k, v in label_vocab.iteritems()}

    return config, max_sent_len, vocab, inv_label_vocab



def showLabels (sent_tok, numvec, inv_label_vocab):
    label_dict = {}

    def add_value (label, value):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(value)

    for i, n in np.ndenumerate(numvec):
        if data_utils.is_ignore_label_num (n):
            continue
        
        label = inv_label_vocab[n]

        if data_utils.is_outside_label(label):
            continue

        pos = i[0] #get index from tuple

        if data_utils.is_begin_label(label):
            add_value (label[2:], sent_tok[pos])
        if data_utils.is_inside_label(label):
            add_value (label[2:], sent_tok[pos])


    return label_dict




def load_model (config):
	#with tf.Graph().as_default():
    model = ElmanRNN (config)

    _ = model.build_model (is_training = False)
    
    train_op, loss = model.build_loss_optimizer (model.outputs)

    init = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init)
    #model.train(session, iterations = 2)

    model.load (session, 'models/model.ckpt')

    return session, model



def predict (session, model, vocab, inv_label_vocab, max_sent_len, sent):
    s = data_utils.basic_tokenizer(sent)
    x = data_utils.sent_to_token_ids(s, vocab, pad_to = max_sent_len)
    print s, x

    probs = model.test(session, np.array([x]) )
    
    probs = probs[0] # single element in batch

    outvec = tf.argmax(probs, dimension=1)
    outnum_vec = outvec.eval(session=session)

    label_dict = showLabels(s, outnum_vec, inv_label_vocab)
    print '->', label_dict #, numvec

    return label_dict

'''

with tf.Graph().as_default():
    model = ElmanRNN (config)
    #model = BiRNN (config)

    _ = model.build_model (is_training = False)
    
    train_op, loss = model.build_loss_optimizer (model.outputs)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        #model.train(session, iterations = 2)

        model.load (session, 'models/model.ckpt')

        for sent in sents:
        	predict (session, model, sent, _inv_label_vocab)

'''









