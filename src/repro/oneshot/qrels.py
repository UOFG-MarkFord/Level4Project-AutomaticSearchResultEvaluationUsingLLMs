import gzip
import json

# Load the query IDs from the cache file
with gzip.open('duot5.cache.json.gz', 'rt') as f:
    cache_data = json.load(f)
query_ids = set(cache_data.keys())

# Load required qrel IDs from DEPdl19.bm25-firstrel.qrels
required_qrel_ids = set()
with open('DEPdl19.bm25-firstrel.qrels', 'r') as f:
    for line in f:
        parts = line.strip().split()
        required_qrel_ids.add(parts[0])

# Read the bm25base file and collect qrels
qrels = {}
with open('dl19-runs/input.bm25base', 'r') as f:
    for line in f:
        parts = line.strip().split()
        query_id, doc_id, score = parts[0], parts[2], float(parts[4])

        if query_id in query_ids and query_id in required_qrel_ids:
            # Assuming higher score indicates higher relevance
            if query_id not in qrels or qrels[query_id]['score'] < score:
                qrels[query_id] = {'doc_id': doc_id, 'score': score}

# Convert qrels to a list and sort by query ID in ascending order
qrels_list = [(query_id, qrel['doc_id']) for query_id, qrel in qrels.items()]
qrels_list.sort(key=lambda x: int(x[0]))  # Sort by query ID

# Limit to the first 42 qrels
qrels_list = qrels_list[:42]

# Write to a file
with open('dl19.bm25-firstrel.qrels', 'w') as f:
    for query_id, doc_id in qrels_list:
        f.write(f"{query_id}\t0\t{doc_id}\t1\n")
