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
#	pprint (label2exprs)
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
				currY.append('O')
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

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def make_vocab (word_lists, normalize_digits):
	for wl in word_lists:
		for w in wl:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

def sent_to_token_ids (sentence, vocab, normalize_digits=True):
	if not normalize_digits:
    	return [vocabulary.get(w, UNK_ID) for w in words]
  	# Normalize digits by 0 before looking words up in the vocabulary.
  	return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


X, Y = get_X_Y_data ()
print zip(X, Y)
