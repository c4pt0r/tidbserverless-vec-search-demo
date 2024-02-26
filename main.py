import optparse
import sys
from db import create_tables, get_tidb_connection, search_embedding_top_n_all
from worker import download_from_urls, get_embedding, watch_feed
from server import serve

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
            # 10 pages a batch, need a worker pool?
            for i in range(0, len(urls), 20):
                download_from_urls(urls[i:i+20])
                print('next batch...')
    if options.query:
        embedding = get_embedding(options.query)
        ret = search_embedding_top_n_all(get_tidb_connection(), embedding)
        for row in ret:
            print(row)
    if options.server:
        addr = options.server.split(':')
        serve(addr[0], int(addr[1]))
    if not options.rss and not options.url_list_file and not options.query and not options.server:
        url = sys.argv[1]
        download_from_urls([url])
        print('Done')
