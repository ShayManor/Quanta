import duckdb
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)

    con = duckdb.connect('data.db')
    con.execute("SELECT title, abstract FROM abstracts")
    results = con.fetchall()
    data = []
    for res in results:
        data.append(res[0] + '\n' + res[1])
    print("Processed data")
    embeddings = model.encode(data).tolist()

    con.execute("CREATE TEMP table temp_emb (idx INTEGER, emb FLOAT[768])")
    con.executemany(
        "INSERT INTO temp_emb VALUES (?, ?)",
        [(i, emb) for i, emb in enumerate(embeddings)]
    )
    con.execute("""
            UPDATE abstracts
            SET embeddings = temp_emb.emb
            FROM temp_emb
            WHERE abstracts.rowid = temp_emb.idx
        """)
    con.commit()

