from wit_to_data import extract_wit_ai_data 
from pprint import pprint



#labels = ['beauty', 'bills','recharge']
labels = ['recharge']

def find_start_tag (tags, start_pos):
    out = [x for x in tags if 'start' in x and  x['start'] == start_pos]
    if len(out) == 0: return None
    assert len(out) == 1
    return out[0]

def get_linear_tags (label2exprs):
    label2tokens = {}

    for label, exprs in label2exprs.viewitems():
        label2tokens[label]  = []
        
        for expr in exprs:
            text = expr['text'].strip()
            tags = expr['tags']

            tokens = []
            curr_token = ''

            i = 0
            while i < len(text):
                c = text[i]
                #print c

                if c.isspace():
                    if curr_token and not curr_token.isspace():
                        tokens.append(curr_token)
                    curr_token = ''
                    i += 1
                else:
                    res = find_start_tag(tags, i)
                    if res != None:
                        end = res['end']
                        entity = res['entity']
                        #print curr_token
                        if curr_token and not curr_token.isspace():
                            tokens.append(curr_token)

                        curr_token = text[i:end]
                        tokens.append( (curr_token, entity) )
                        curr_token = ''
                        i = end
                    else:
                        curr_token = curr_token + c
                        i += 1

            print text, tokens
            label2tokens[label].append({'text': text, 'tokens': tokens})

    return label2tokens
                    



#returns X = [list of words] Y = [ list of labels ]
def get_X_Y_data ():
    label2exprs = extract_wit_ai_data(labels)
#   pprint (label2exprs)
    label2tokens = get_linear_tags(label2exprs)

    sents = label2tokens['recharge']

    X = []
    Y = []

    for sent in sents:
        tokens = sent['tokens']
        currX = []
        currY = []
        for tok in tokens:

            if isinstance(tok, basestring):
                currX.append(tok)
                currY.append('__O__')
            else:
                words, label = tok
                ws = words.split(' ')
                for i, w in enumerate(ws):
                    currX.append(w)
                    if i == 0: pr = 'B_'
                    else: pr = 'I_'
                    currY.append(pr + label)


        X.append(currX)
        Y.append(currY)

    return X, Y


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

    sent_lens = {}
    for s in X:
        slen = len(s)
        if not slen in sent_lens:
            sent_lens[slen] = 0
        sent_lens[slen] = sent_lens[slen] + 1

    max_sent_len = max([len(s) for s in X])

    print max_sent_len, sent_lens


    X_num = []
    Y_num = []
    #print zip(X, Y)

    vocab, rev_vocab = make_vocab (X + Y)
    print vocab


    for x, y in zip(X, Y):
        x_toks = sent_to_token_ids(x, vocab, pad_to = max_sent_len)
        y_toks = sent_to_token_ids(y, vocab, pad_to = max_sent_len)
        X_num.append(x_toks)
        Y_num.append(y_toks)

    gen = ( z for z in zip(X_num, Y_num) )
    return gen, max_sent_len, len(vocab)
    #return X_num, Y_num

def get_next_batch (gen, batch_size):
    return next(gen)

gen, _, _ = get_all_data()


'''

while True:
    #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
    
    try:
        z = get_next_batch (gen, 1)
        print z
    except:
        break
'''



