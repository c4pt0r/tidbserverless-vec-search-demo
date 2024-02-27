# tidbserverless-vec-search-demo
a small demo

```
python3 -m venv . 
source bin/activate
pip install -r requirements.txt
python main.py -h

# Prepare .env

cat << EOF > .env
DATABASE_TIDB_HOST=127.0.0.1
DATABASE_TIDB_NAME=tidb_vec_demo_dongxu
EOF

# Prepare dataset

python main.py -u <url-list-file>

# Run server

python main.py -s 0.0.0.0:5001
```
