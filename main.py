import time
import optparse
import hashlib
import feedparser
import newspaper
from bs4 import BeautifulSoup
from newspaper import Article
from mattsollamatools import chunker
from sentence_transformers import SentenceTransformer
from db import get_tidb_connection, create_tables, generate_id
from db import insert_articles, insert_chunks, insert_embeddings, search_embedding_top_n_all, get_chunk_by_ids, filter_existed_url_hashes 
from db import ArticleModel, ChunkModel, EmbeddingModel

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def hash(text : str) -> str:
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    return md5.hexdigest()

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
        text = url2text_artical(url)
        article_id = generate_id()
        #TODO: get related links
        related_links = [url]
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

def url2text_artical(url) -> str:
    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'}
    article = Article(url, headers=headers)
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        return ''
    lines = [t.strip() for t in article.text.split('\n')]
    text = '\n'.join([t for t in lines if t])
    # remove html tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator='\n')
    return text

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

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-r', '--rss', dest='url', help='RSS URL')
    parser.add_option('-q', '--query', dest='query', help='Query')
    options, args = parser.parse_args()

    create_tables()
    if options.url:
        watch_feed(options.url)
    if options.query:
        embedding = get_embedding(options.query)
        ret = search_embedding_top_n_all(get_tidb_connection(), embedding)
        for row in ret:
            print(row)
