import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from easydict import EasyDict as edict

import data_utils
import json



from model_rnn import Sent2Labels
from easydict import EasyDict as edict


with open('models/config.json', 'r') as f:
	config = json.load(f)
	config = edict(config)
	print config

max_sent_len = config['num_steps']

with open('models/vocab.json', 'r') as f:
	vocab = json.load(f)
	#vocab = edict(vocab)
	print vocab

sents = [
	'I want to recharge',
	'recharge my phone for 100 rs',
	'recharge 9988776655 for rs 100',
	'rs 50',
	'Can I get Rs 200 recharge for 9912312345',
	'pay my postpaid bill'
]




with tf.Graph().as_default():
    model = Sent2Labels (config)

    _, _ = model.build_model (is_training = False)
    
    train_op, loss = model.build_loss_optimizer (model.outputs)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        #model.train(session, iterations = 2)

        model.load (session, 'models/model.ckpt')

        for sent in sents:
			s = sent.split()
			x = data_utils.sent_to_token_ids(s, vocab, pad_to = max_sent_len)
			print s, x

			probs = model.test(session, np.array([x]) )
			
			probs = probs[0] # single element in batch

			outvec = tf.argmax(probs, dimension=1)
			print outvec.eval()








