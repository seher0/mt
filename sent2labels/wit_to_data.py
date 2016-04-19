
import json
from pprint import pprint



labels = ['beauty', 'bills','recharge']


def extract_wit_ai_data (labels, PREFIX = '../wit-ai/'):

	label2exprs = {}

	for i, label in enumerate(labels):
		fname = PREFIX + label + '.json'
		with open(fname) as f:
			print fname
			eid2name = {}

			data = json.load(f)
			entities = data['entities']
			exprs = data['expressions']

			if len(entities) == 0 : continue

			for e in entities:
				name = e['name']
				eid = e['id']
				eid2name[eid] = name

			new_exprs = []
			for ex in exprs:
				text = ex['body']
				try:
					ex_ents = ex['entities']
					for ex_ent in ex_ents:
						ex_ent['tagname'] = eid2name[ ex_ent[u'wisp']]
						ex_ent.pop( 'wisp', None)

					new_exprs.append ( { 
						'text': text.strip().lower(),
						'tags': ex_ents
					})
				except:
					print ex
					continue

			label2exprs[label] = new_exprs
			return label2exprs


label2exprs = extract_wit_ai_data(labels)


pprint (label2exprs)
