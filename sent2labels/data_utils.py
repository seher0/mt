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
					









def getData():
	label2exprs = extract_wit_ai_data(labels)
#	pprint (label2exprs)
	label2tokens = get_linear_tags(label2exprs)

	#pprint(label2tokens)



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

X, Y = get_X_Y_data ()
print zip(X, Y)
