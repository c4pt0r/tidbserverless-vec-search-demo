import os
import time
import MySQLdb
import json
from snowflake import SnowflakeGenerator

from pydantic import BaseModel
from typing import Optional, List
from pydantic.types import Json

from dotenv import load_dotenv
load_dotenv()

DATABASE_TIDB_HOST = os.getenv('DATABASE_TIDB_HOST', '')
DATABASE_TIDB_PORT = int(os.getenv('DATABASE_TIDB_PORT', '4000'))
DATABASE_TIDB_USER = os.getenv('DATABASE_TIDB_USER', 'root')
DATABASE_TIDB_PASSWORD = os.getenv('DATABASE_TIDB_PASSWORD', '')
DATABASE_TIDB_NAME = os.getenv('DATABASE_TIDB_NAME', 'hnsummary')
DATABASE_TIDB_SSL_CA = os.getenv('DATABASE_TIDB_SSL_CA', '')

def get_tidb_connection():
    connection = MySQLdb.connect(
            host=DATABASE_TIDB_HOST,
            port=int(DATABASE_TIDB_PORT),
            user=DATABASE_TIDB_USER,
            password=DATABASE_TIDB_PASSWORD,
            database=DATABASE_TIDB_NAME,
            #ssl_mode="VERIFY_IDENTITY",
            autocommit=True,
            charset='utf8mb4',
            #ssl={"ca": DATABASE_TIDB_SSL_CA}
            )
    return connection

_id_gen = SnowflakeGenerator(0)
def generate_id():
    return next(_id_gen)

class ArticleModel(BaseModel):
    article_id: Optional[int] = None
    title: str
    content: Optional[str] = None
    source_url: str
    url_hash: str
    related_links: Optional[list] = None

    @classmethod
    def sql(self):
        return '''
        CREATE TABLE IF NOT EXISTS articles (
            article_id BIGINT UNSIGNED PRIMARY KEY,
            title TEXT NOT NULL,
            content_description LONGTEXT,
            source_url TEXT NOT NULL,
            url_hash VARCHAR(256) NOT NULL UNIQUE KEY,
            related_links JSON,
            create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''

class ChunkModel(BaseModel):
    chunk_id: Optional[int] = None
    article_id: int
    chunk_text: str

    @classmethod
    def sql(self):
        return '''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id BIGINT UNSIGNED PRIMARY KEY,
            article_id BIGINT UNSIGNED NOT NULL,
            chunk_text LONGTEXT NOT NULL,
            create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''

class EmbeddingModel(BaseModel):
    embedding_id: Optional[int] = None
    chunk_id: int
    embedding_data: List[float]

    @classmethod
    def sql(self):
        return '''
        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id BIGINT UNSIGNED PRIMARY KEY,
            chunk_id BIGINT UNSIGNED NOT NULL,
            embeddings VECTOR NOT NULL,
            create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    '''

def create_tables():
    conn = get_tidb_connection()
    cursor = conn.cursor()
    cursor.execute(ArticleModel.sql())
    cursor.execute(ChunkModel.sql())
    cursor.execute(EmbeddingModel.sql())
    cursor.close()
    conn.close()

def filter_existed_url_hashes(conn, url_hash_list: list[str]):
    val = ','.join(['%s'] * len(url_hash_list))
    sql = f'''SELECT url_hash
        FROM articles
        WHERE url_hash in ({val})
        '''
    cursor = conn.cursor()
    cursor.execute(sql, tuple(url_hash_list))
    result = cursor.fetchall()
    exists_hashs = []
    for row in result:
        exists_hashs.append(row[0])
    cursor.close()
    return exists_hashs

def insert_articles(conn, articles : list[ArticleModel]):
    insert_sql = '''
            INSERT IGNORE INTO articles (article_id, title, content_description, source_url, url_hash, related_links)
            VALUES (%s, %s, %s, %s, %s, %s);
        '''
    with conn.cursor() as cursor:
        values = []
        for article in articles:
            values.append((article.article_id, article.title, article.content, article.source_url, article.url_hash, json.dumps(article.related_links)))
        try:
            cursor.executemany(insert_sql, values)
            conn.commit()
        except MySQLdb.IntegrityError as e:
            if 'Duplicate entry' in str(e):
                print('Duplicate entry')
                return

def insert_chunks(conn, chunks : list[ChunkModel]):
    insert_sql = '''
            INSERT INTO chunks (chunk_id, article_id, chunk_text)
            VALUES (%s, %s, %s);
        '''
    with conn.cursor() as cursor:
        values = []
        for chunk in chunks:
            values.append((chunk.chunk_id, chunk.article_id, chunk.chunk_text))
        cursor.executemany(insert_sql, values)
        conn.commit()

def insert_embeddings(conn, embeddings : list[EmbeddingModel]):
    insert_sql = '''
            INSERT INTO embeddings (embedding_id, chunk_id, embeddings)
            VALUES (%s, %s, %s);
        '''
    with conn.cursor() as cursor:
        values = []
        for embedding in embeddings:
            values.append((embedding.embedding_id, embedding.chunk_id, str(embedding.embedding_data)))
        cursor.executemany(insert_sql, values)
        conn.commit()

def search_embedding_top_n(conn, embedding, n = 10):
    search_sql = '''
        SELECT embeddings,chunk_id, 
            VEC_Cosine_Distance(embeddings, %s) AS d
        FROM embeddings
        ORDER BY d LIMIT %s;
        '''
    cursor = conn.cursor()
    cursor.execute(search_sql, (str(embedding), n))
    result = cursor.fetchall()
    ret = []
    for row in result:
        chunk_id = row[1]
        distance = row[2]
        ret.append((chunk_id, distance))
    cursor.close()
    return ret

def search_embedding_top_n_all(conn, embeddings, n = 10):
    search_sql = '''
        SELECT e.embeddings, 
               c.chunk_id, 
               c.chunk_text, 
               c.article_id,
               a.title,
               a.source_url,
               VEC_Cosine_Distance(e.embeddings, %s) AS d
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
        JOIN articles a ON c.article_id = a.article_id
        AND LENGTH(c.chunk_text) > 50
        ORDER BY d ASC
        LIMIT %s
    '''
    cursor = conn.cursor()
    cursor.execute(search_sql, (str(embeddings), n))
    result = cursor.fetchall()
    ret = []
    for row in result:
        chunk_id = row[1]
        chunk_text = row[2]
        article_id = row[3]
        title = row[4]
        url = row[5]
        distance = row[6]
        ret.append({
            'chunk_id': chunk_id,
            'content': chunk_text,
            'article_id': article_id,
            'distance': distance,
            'title': title,
            'url': url
        })
    cursor.close()
    return ret

def get_chunk_by_ids(conn, chunk_ids):
    search_sql = '''
        SELECT chunk_text, article_id
        FROM chunks
        WHERE chunk_id in (%s);
        '''
    cursor = conn.cursor()
    val = ','.join(['%s'] * len(chunk_ids))
    cursor.execute(search_sql % val, tuple(chunk_ids))
    results = cursor.fetchall()
    ret = []
    for row in results:
        ret.append((row[0], row[1]))
    cursor.close()
    return ret
