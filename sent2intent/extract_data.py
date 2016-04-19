
import json

def extract_api_ai_data (labels, PREFIX = '../api-ai/intents/'):
	label2text = {}

	def getText (d):
		#print 'in getText'
		texts = []

		for item in d['userSays']:
			d = item['data']
			#assert len(d) == 1
			t = d[0]['text'].strip().lower()
			#t = t.encode('unicode')
			if '@' in t:
				continue
			#print t

			texts.append ( t )

		return texts
	
	MAX = 0

	for i, label in enumerate(labels):
		
		fname = PREFIX + label + '.json'
		with open(fname) as f:
			#try:
			#print 'label is **' + label
			data = json.load(f)
			texts = getText(data)
			#print len(texts)
			label2text[label] = texts
			print label, texts, '\n'

			max_ = max( len(s) for s in texts)
			if (max_ > MAX): 
				MAX = max_
		
			#except Exception as e:
			#	print 'exception **', e

	return label2text, MAX

labels = ['beauty', 'bills', 'grocery', 'prescription', 'recharge', 'Refunds', 'Referral', 'support', 'offers']


extract_api_ai_data (labels)