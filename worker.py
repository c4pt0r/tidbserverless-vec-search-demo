import time
import hashlib
import feedparser
import newspaper
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from newspaper import Article
from mattsollamatools import chunker
from sentence_transformers import SentenceTransformer
from db import get_tidb_connection, generate_id
from db import insert_articles, insert_chunks, insert_embeddings, filter_existed_url_hashes
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

class Task:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.status = 'pending'
        self.mu = threading.Lock()

    def update_status(self, status):
        with self.mu:
            self.status = status
    
    def get_status(self):
        with self.mu:
            return self.status
    
    def get_result(self):
        with self.mu:
            return self.result

    def set_result(self, result):
        with self.mu:
            self.result = result
 
    def run(self):
        self.update_status('running')
        try:
            ret = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.set_result(e)
            self.update_status('error')
            return
        self.set_result(ret)
        self.update_status('done')

class DownloadTask(Task):
    def __init__(self, url):
        super().__init__(url2text_artical, url)
        self.url = url


class WorkerPool:
    def __init__(self, max_workers = 0):
        if max_workers == 0:
            max_workers = os.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_task(self, task):
        self.executor.submit(task.run)

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def submit_tasks_and_wait(self, tasks):
        for task in tasks:
            self.submit_task(task)
        for task in tasks:
            while task.get_status() != 'done' and task.get_status() != 'error':
                time.sleep(0.1)
        self.shutdown()
        return [task.get_result() for task in tasks if task.get_status() == 'done']

def download_from_urls_parallel(urls: list) -> (list[ArticleModel], list):
    error_list = []
    tasks = []
    for url in urls:
        tasks.append(DownloadTask(url))
    pool = WorkerPool()
    pool.submit_tasks_and_wait(tasks)
    # collect results
    new_articles = []
    for task in tasks:
        if task.get_status() == 'done':
            text, links = task.get_result()
            url = task.url
            article_id = generate_id()
            article = ArticleModel(article_id=article_id,
                                    title=url,
                                    content=text,
                                    source_url=url,
                                    url_hash=hash(url),
                                    related_links=links + [url])
            new_articles.append(article)
        elif task.get_status() == 'error':
            error_list.append(task.get_result())
    return new_articles, error_list

def insert_articles_and_embeddings_parallel(articles: list[ArticleModel]) -> list:
    pool = WorkerPool()
    ret = []
    tasks = []
    for article in articles:
        t = Task(insert_articles_and_embeddings, [article])
        pool.submit_task(t)
        tasks.append(t)
    # collect results
    for task in tasks:
        while task.get_status() != 'done' and task.get_status() != 'error':
            time.sleep(0.1)
    pool.shutdown()
    return [r.get_result() for r in tasks if r.get_status() == 'done']

def insert_articles_and_embeddings(articles: list[ArticleModel]) -> list:
    conn = get_tidb_connection()
    # first insert articles
    article_ids = []
    insert_articles(conn, articles)
    # build chunks and embeddings
    for article in articles:
        chunk_modles = []
        embedding_models = []
        embeddings, chunks = get_chunks_embeddings(article.title + '\n' + article.content)
        for i, chunk_text in enumerate(chunks):
            chunk_id = generate_id()
            vec = embeddings[i].tolist()
            chunk_model = ChunkModel(article_id=article.article_id,
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
        article_ids.append(article.article_id)
    conn.close()
    return article_ids

def download_from_urls(urls: list) -> None:
    new_articles = []
    for url in urls:
        text, links = url2text_artical(url)
        if links is None:
            links = []
        article_id = generate_id()
        article = ArticleModel(article_id=article_id,
                                title=url,
                                content=text,
                                source_url=url,
                                url_hash=hash(url),
                                related_links=links + [url])
        new_articles.append(article)
    return new_articles

def get_new_urls(conn, url_list: list[str]) -> list[str]:
    url_hash = [hash(url) for url in url_list]
    exists = filter_existed_url_hashes(conn, url_hash)
    return [url for url, h in zip(url_list, url_hash) if h not in exists]

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