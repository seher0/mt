
import json
from pprint import pprint


labels = ['beauty', 'bills', 'grocery', 'prescription', 'recharge', 'Refunds', 'Referral', 'support', 'offers']

from extract_data import extract_api_ai_data



label2text, max_sent_length = extract_api_ai_data(labels)



def analyze(sent):
    tokens = nlp(sent)
    print '^^ ', sent
    res = []
    for tok in tokens:
        
        print tok, tok.tag_, (tok.dep_, tok.head)
        if tok.dep_ == 'dobj':
            #print "->" , tok
            res.append(tok)
            for tok1 in tokens:
                if tok1.head is tok:
                    if tok1.tag_ not in ['PRP', 'PRP$', 'DT']:
                        res.append(tok1)
            h = tok.head
            if h.dep_ not in [u'ROOT',u'xcomp']:
                #print ('##3 h.dep_', h.dep_)
                res.append(h)
    
    if len(res) > 0:
        print '--> res is ', res
        return res
    
    for tok in tokens:
        #print tok, tok.tag_, (tok.dep_, tok.head)
        if tok.dep_ == 'amod':
            res.append(tok)
    
    if len(res) > 0:
        print '--> res is ', res
        return res
        
        
    #for chunk in tokens.noun_chunks:
    #    print chunk.orth_

#analyze(u'I want a recharge')  
analyze(u'book facial for me')

for label in label2text:
    texts = label2text[label]
    #if label is not 'Refunds':
    #    continue
    print "** " + label
    for sent in texts:
        analyze(sent)
        print '--'