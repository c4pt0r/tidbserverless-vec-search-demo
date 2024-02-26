from flask import Flask, request, render_template
from worker import download_from_urls, get_embedding
from db import get_tidb_connection, search_embedding_top_n_all

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        embedding = get_embedding(query)
        results = search_embedding_top_n_all(get_tidb_connection(), embedding)
        return render_template('index.html', results=results, query=query)
    return render_template('index.html', results=None, query=None)

@app.route('/admin' , methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        url = request.form['url']
        download_from_urls([url])
        return render_template('admin.html', message='Done')
    return render_template('admin.html', message=None)

def serve(host, port):
    app.run(host=host, port=port, debug=True)
