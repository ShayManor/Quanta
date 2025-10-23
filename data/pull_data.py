import json
import os

import duckdb

from src.PaperDataset import PaperDataset

length = 1000
dataset = PaperDataset("jackkuo/arXiv-metadata-oai-snapshot", length, split="train", cache_size=length)
data = []
for idx in range(dataset.length):
    data.append(dataset[idx])

with open('data.json', "w") as f:
    f.writelines(json.dumps(data))

con = duckdb.connect('data.db')
con.execute("CREATE TABLE IF NOT EXISTS abstracts AS SELECT * FROM read_json_auto('data.json');")
con.execute("ALTER TABLE abstracts ADD COLUMN embeddings FLOAT[768];")
con.close()
os.remove('data.json')
