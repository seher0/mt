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
	for label, exprs in label2exprs.viewitems():
		
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
						label = res['entity']
						#print curr_token
						if curr_token and not curr_token.isspace():
							tokens.append(curr_token)

						curr_token = text[i:end]
						tokens.append( (curr_token, label) )
						curr_token = ''
						i = end
					else:
						curr_token = curr_token + c
						i += 1

			print text, tokens


					







def main():
	label2exprs = extract_wit_ai_data(labels)
#	pprint (label2exprs)
	get_linear_tags(label2exprs)



main()

