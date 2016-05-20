import sys

sys.path.append('../learn-dialog/data')


from wit_to_train_data import get_linear_tags 
from pprint import pprint


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#labels = ['beauty', 'bills','recharge']
labels = ['recharge', 'cab', 'bus' ]



label2tokens = get_linear_tags(labels, PREFIX='../learn-dialog/data')
#pprint (label2tokens)
                    
BEGIN_PREFIX = 'B_'
INSIDE_PREFIX = 'I_'
OUTSIDE_LABEL = '__O__'

def is_begin_label (lab):
    return lab.startswith(BEGIN_PREFIX)

def is_inside_label (lab):
    return lab.startswith(INSIDE_PREFIX)

def is_outside_label (lab):
    return lab.startswith(OUTSIDE_LABEL)

def is_ignore_label_num (label_num):
    return label_num >= PAD_ID and label_num <= UNK_ID

#returns X = [list of words] Y = [ list of labels ]
def get_X_Y_data ():
#   pprint (label2exprs)
    global label2tokens


    X = []
    Y = []


    for category, sents in label2tokens.iteritems():

        #sents = label2tokens['recharge']

        for sent in sents:
            tokens = sent['tokens']
            currX = []
            currY = []
            for tok in tokens:

                if isinstance(tok, basestring):
                    currX.append(tok)
                    currY.append(OUTSIDE_LABEL)
                else:
                    words, label = tok
                    ws = words.split(' ')
                    for i, w in enumerate(ws):
                        currX.append(w)
                        if i == 0: pr = BEGIN_PREFIX
                        else: pr = INSIDE_PREFIX
                        currY.append(pr + label)


            X.append(currX)
            Y.append(currY)

    return X, Y



import re

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def make_vocab (word_lists, max_vocabulary_size=100000, normalize_digits=True):
    vocab_map = {}
    for wl in word_lists:
        for w in wl:
            word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
            if word in vocab_map:
                vocab_map[word] += 1
            else:
                vocab_map[word] = 1
      
    vocab_list = _START_VOCAB + sorted(vocab_map, key=vocab_map.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

    '''
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + b"\n")
    '''
    vocab = {}
    rev_vocab = {}
    for i, word in enumerate(vocab_list):
        vocab[word.strip()] = i
        rev_vocab[i] = word.strip()

    return vocab, rev_vocab

def sent_to_token_ids (sentence, vocabulary, pad_to = None, normalize_digits=True):
    if not normalize_digits:
        ret = [vocabulary.get(w, UNK_ID) for w in sentence]
    # Normalize digits by 0 before looking words up in the vocabulary.
    ret = [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in sentence]

    if pad_to:
        pad_len = pad_to - len(ret)
        ret = ret + [PAD_ID] * pad_len
    return ret

def get_all_data():
    X, Y = get_X_Y_data ()

    #pprint (zip(X, Y))

    # find max sent length
    sent_lens = {}
    for s in X:
        slen = len(s)
        if not slen in sent_lens:
            sent_lens[slen] = 0
        sent_lens[slen] = sent_lens[slen] + 1

    max_sent_len = max([len(s) for s in X])

    print max_sent_len, sent_lens


    #convert strings to token ids with padding
    X_num = []
    Y_num = []
    #print zip(X, Y)

    # generate vocabulary
    vocab, rev_vocab = make_vocab (X)
    label_vocab, _ = make_vocab (Y)
    print vocab
    print label_vocab


    for x, y in zip(X, Y):
        x_toks = sent_to_token_ids(x, vocab, pad_to = max_sent_len)
        y_toks = sent_to_token_ids(y, label_vocab, pad_to = max_sent_len)
        X_num.append(x_toks)
        Y_num.append(y_toks)

    gen = ( z for z in zip(X_num, Y_num) )
    return gen, max_sent_len, vocab, label_vocab
    #return X_num, Y_num

def get_next_batch (gen, batch_size):
    return next(gen)

#gen, _, _, _ = get_all_data()


'''

while True:
    #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
    
    try:
        z = get_next_batch (gen, 1)
        print z
    except:
        break
'''



