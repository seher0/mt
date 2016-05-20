from bottle import route, run
import json
from test import init, load_model, predict

config, max_sent_len, vocab, inv_label_vocab = init()

session, model = load_model(config)

@route('/getEntities/<sent>')
def index(sent):
    from bottle import response

    #try:
    output = predict (session, model, vocab, inv_label_vocab, max_sent_len, sent)

    response.content_type = 'application/json'
    ''' 
    Check if this is a good or bad response. 
    '''
    return json.dumps(output)
    #except ValueError:
    #    print "Error"
    #    return ''


run(host='localhost', port=7760, reloader=True)
