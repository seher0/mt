import json
from pprint import pprint


labels = ['beauty', 'bills', 'grocery', 'prescription', 'recharge', 'Refunds', 'Referral', 'support', 'offers']

from extract_data import extract_api_ai_data


NUM_CHARS = 255







import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils

def char2int(c):
	return ord(c)

def sent_to_char_seq(sent, MAX_LENGTH):
    sent_chars = list(sent)
    sent_chars_indices = map(lambda c: char2int(c), sent_chars)
    #print sent_chars_indices
    sent_chars_indices = sent_chars_indices + [0]*(MAX_LENGTH - len(sent_chars_indices))
    return sent_chars_indices
    #return sequence.pad_sequences([sent_chars_indices], maxlen=MAX_LENGTH)[0]


def build_X_Y_set (label2text, max_sent_length):
	X = []
	Y = []

	Xsent = []
	Ylab = []

	for label in label2text:
		label_index = labels.index(label)

		texts = label2text[label]
		for sent in texts:
			sent_seq = sent_to_char_seq (sent, max_sent_length)

			X.append(sent_seq)
			Y.append(label_index)
			Xsent.append(sent)
			Ylab.append(label)


	X = np.array(X).astype(np.uint8)
	Y = np_utils.to_categorical(np.array(Y)).astype(np.bool)

	print(X.shape, Y.shape)

	return X, Y, Xsent, Ylab


def build_model (MAX_LENGTH, num_chars, num_labels):



	model = Sequential()
	model.add(Embedding(num_chars, 16, input_length=MAX_LENGTH, mask_zero=True))
	model.add(LSTM(output_dim = 32, init='glorot_uniform', inner_init='orthogonal',
	               activation='tanh', inner_activation='hard_sigmoid', 
	               return_sequences=False))
	model.add(Dropout(0.5))
	model.add(Dense(num_labels))
	model.add(Activation('softmax'))

	print 'compiling model'

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop') #, mode='categorical')

	print 'model compiled'
	return model

	

def train (model, X, Y):
	#from sklearn.cross_validation import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	#train =[X_train, y_train]
	#test = [X_test, y_test]

	print 'start train'
	X_train = X
	Y_train = Y

	batch_size = 1
	nb_epoch = 150


	early_stopping = EarlyStopping(patience=5, verbose=1)
	checkpointer = ModelCheckpoint(filepath='sent2intent_keras_weights.hdf5', 
	                               verbose=1, 
	                               save_best_only=True)

	print ('fitting model')

	model.fit(X_train, Y_train, 
	          batch_size=batch_size, 
	          nb_epoch=nb_epoch, 
	          show_accuracy=True,
	          verbose=1,
	          shuffle=True,
	          validation_split=0.1,
	          callbacks=[ checkpointer])



def predict (model, X_test, Y_test):
	from sklearn.metrics import confusion_matrix, classification_report

	model.load_weights('sent2intent_keras_weights.hdf5')
	preds = model.predict_classes(X_test, batch_size=64, verbose=0)

	print('')
	print(classification_report(np.argmax(Y_test, axis=1), preds, target_names=labels))
	print('')
	print(confusion_matrix(np.argmax(Y_test, axis=1), preds))


def test (model, sent, max_sent_length):
	X = sent_to_char_seq(sent, max_sent_length)
	X = np.array([X]).astype(np.uint8)
	model.load_weights('sent2intent_keras_weights.hdf5')
	preds = model.predict_classes(X, batch_size=64, verbose=0)
	print labels
	print preds


label2text, max_sent_length = extract_api_ai_data(labels)
#pprint (label2text)

X, Y, Xsent, Ylab = build_X_Y_set (label2text, max_sent_length)
#print Y

model = build_model (max_sent_length, NUM_CHARS, len(labels))

#train (model, X, Y)

#predict(model, X, Y)

test(model, 'please recharge', max_sent_length)
test(model, 'recharge my phone', max_sent_length)
test(model, 'i wanna recharge', max_sent_length)
test(model, 'How can I recharge from this app', max_sent_length)











