import time
import optparse
import hashlib
import feedparser
import newspaper
from bs4 import BeautifulSoup
from newspaper import Article
from mattsollamatools import chunker
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request
from db import get_tidb_connection, create_tables, generate_id
from db import insert_articles, insert_chunks, insert_embeddings, search_embedding_top_n_all, get_chunk_by_ids, filter_existed_url_hashes 
from db import ArticleModel, ChunkModel, EmbeddingModel

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def get_links_from_html(html: str) -> list:
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        links.append(a['href'])
    return links

def hash(text : str) -> str:
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    return md5.hexdigest()

def download_from_urls(urls: list) -> None:
    conn = get_tidb_connection()
    url_hashes = []
    for url in urls:
        url_hashes.append(hash(url))
    exists_hashes = filter_existed_url_hashes(conn, url_hashes)

    new_articles = []
    for url in urls:
        if hash(url) not in exists_hashes:
            text, links = url2text_artical(url)
            article_id = generate_id()
            # insert new articles
            article = ArticleModel(article_id=article_id,
                                    title=url,
                                    content=text,
                                    source_url=url,
                                    url_hash=hash(url),
                                    related_links=links + [url])
            new_articles.append(article)
    if len(new_articles) > 0:
        print('found new articles, inserting...')
        insert_articles(conn, new_articles)
    else:
        print('No new articles found')
        return
    # build chunks and embeddings
    for article in new_articles:
        chunk_modles = []
        embedding_models = []
        embeddings, chunks = get_chunks_embeddings(article.content)
        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_id()
            vec = embeddings[i].tolist()
            chunk_model = ChunkModel(article_id=article_id,
                                     chunk_id=chunk_id,
                                     chunk_text=chunk_text)
            embedding_model = EmbeddingModel(
                embedding_id=generate_id(),
                chunk_id=chunk_id,
                embedding_data=vec
            )
            chunk_modles.append(chunk_model)
            embedding_models.append(embedding_model)
        insert_chunks(conn, chunk_modles)
        insert_embeddings(conn, embedding_models)
    conn.close()
    print('Done')

def download_from_rss(rss_url: str) -> None:
    feed = feedparser.parse(rss_url)
    conn = get_tidb_connection()
    url_hashes = []
    for entry in feed.entries:
        url_hashes.append(hash(entry.link))
    exists_hashes = filter_existed_url_hashes(conn, url_hashes)

    entries = []
    for entry in feed.entries:
        if hash(entry.link) not in exists_hashes:
            entries.append(entry)

    # insert new articles
    if len(entries) > 0:
        print(f'Found new articles, inserting {len(entries)} articles')
    else:
        print('No new articles found')
        return
    # check url hash if it's in db
    new_articles = []
    for entry in entries:
        url = entry.link
        text, links = url2text_artical(url)
        article_id = generate_id()
        #TODO: get related links
        related_links = [url] + links
        article = ArticleModel(article_id=article_id,
                                 title=entry.title,
                                 content=text,
                                 source_url=url,
                                 url_hash=hash(url),
                                 related_links=related_links)
        new_articles.append(article)

    insert_articles(conn, new_articles)
    # build chunks and embeddings
    for article in new_articles:
        chunk_modles = []
        embedding_models = []
        embeddings, chunks = get_chunks_embeddings(article.title + '\n' + article.content)
        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_id()
            vec = embeddings[i].tolist()
            chunk_model = ChunkModel(article_id=article_id,
                                        chunk_id=chunk_id,
                                        chunk_text=chunk_text)
            embedding_model = EmbeddingModel(
                                        embedding_id=generate_id(),
                                        chunk_id=chunk_id,
                                        embedding_data=vec
                                    )
            chunk_modles.append(chunk_model)
            embedding_models.append(embedding_model)
        insert_chunks(conn, chunk_modles)
        insert_embeddings(conn, embedding_models)
    conn.close()

def url2text_artical(url) -> (str, list):
    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'}
    article = Article(url, headers=headers)
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        return '', None
    lines = [t.strip() for t in article.text.split('\n')]
    text = '\n'.join([t for t in lines if t])
    # remove html tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator='\n')
    # get links
    links = get_links_from_html(article.html)
    return text, links

# Split text into chunks and generate embeddings for each chunk
def get_chunks_embeddings(text: str) -> (list[list[float]], list):
    chunks = chunker(text)
    embeddings = model.encode(chunks)
    return embeddings, chunks

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()

def watch_feed(url):
    while True:
        print('start checking rss...')
        download_from_rss(url)
        time.sleep(10)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        embedding = model.encode(query).tolist()
        results = search_embedding_top_n_all(get_tidb_connection(), embedding)
        return render_template('index.html', results=results, query=query)
    return render_template('index.html', results=None, query=None)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-r', '--rss', dest='rss', help='RSS URL')
    parser.add_option('-u', '--url-list-file', dest='url_list_file', help='URL list file')
    parser.add_option('-q', '--query', dest='query', help='Query')
    parser.add_option('-s', '--server', dest='server', help='Server')

    options, args = parser.parse_args()

    create_tables()
    if options.rss:
        watch_feed(options.url)
    if options.url_list_file:
        with open(options.url_list_file, 'r') as f:
            urls = f.readlines()
            download_from_urls(urls)
    if options.query:
        embedding = get_embedding(options.query)
        ret = search_embedding_top_n_all(get_tidb_connection(), embedding)
        for row in ret:
            print(row)
    if options.server:
        addr = options.server.split(':')
        app.run(host=addr[0], port=int(addr[1]), debug=True)
