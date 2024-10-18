from flask import Flask, request
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

@app.route('/', methods=['POST'])
def status():
    # get field 'message' from the form
    status = request.form['message']
    print(status)
    return 'OK'

@app.route('/', methods=['GET'])
def hello():
    return 'Hello'

port = 9031
if __name__ == '__main__':
    server = WSGIServer(('0.0.0.0', port), app, log=None)
    print('Server started on port %d' % port)
    server.serve_forever()
