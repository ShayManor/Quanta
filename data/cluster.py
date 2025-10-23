from typing import List, Optional

import duckdb
import numpy as np


def cluster_embeddings(embs):
    """
    Takes embeddings, clusters and returns the clusters
    :param embs: list of all available embeddings in latent space
    :return: list^2 of embeddings corresponding to clusters
    """
    return [embs[:len(embs) // 2], embs[len(embs) // 2:]]


def get_avg(embs):
    """
    Gets average embedding of cluster for comparison
    :param embs: list of embeddings in the cluster - may be sampled
    :return: normalized embeddings
    """
    # if len(embs) > 100:
    #     embs = np.random.choice(embs, size=100, replace=False)

    return np.mean(embs, axis=0)[0].tolist()


def cluster(location: Optional[List[int]], con, max_cluster_size=10, min_cluster_size=1):
    """
    Recursively clusters embeddings into tree
    :param location: path (array of ints) to current location
    :param con: duckdb client
    :param max_cluster_size: keeps splitting until less than max_cluster_size
    :param min_cluster_size: minimum size for cluster, > 1 could drop data.
    """
    print(location)
    if not location:
        con.execute("SELECT embeddings FROM abstracts;")
        embeddings = con.fetchall()
    else:
        con.execute("SELECT embeddings FROM abstracts WHERE path = ?;", [location])
        embeddings = con.fetchall()

    cluster_size = len(embeddings)
    if cluster_size == 0:
        print(embeddings)
    print(f"Cluster size: {cluster_size}")
    if cluster_size < min_cluster_size:
        # Too small cluster - drop
        return
    if cluster_size <= max_cluster_size:
        # Finished cluster
        return

    clusters = cluster_embeddings(embeddings)
    con.execute("INSERT INTO tree VALUES (?, ?);", [get_avg(embeddings), location if location else [0]])
    for idx, current_cluster in enumerate(clusters):
        # Append index to path
        con.execute("CREATE TEMP TABLE IF NOT EXISTS cluster_temp (emb FLOAT[768])")
        con.executemany("INSERT INTO cluster_temp VALUES (?)", [(e[0],) for e in current_cluster])
        con.execute(
            """
            UPDATE abstracts
            SET path = list_append(COALESCE(path, []), ?)
            FROM cluster_temp
            WHERE abstracts.embeddings = cluster_temp.emb
        """,
            [idx],
        )
        con.execute("DELETE FROM cluster_temp")

        current_path = location + [idx] if location else [idx]

        con.execute("INSERT INTO tree VALUES (?, ?);", [get_avg(current_cluster), current_path])
        con.commit()
        cluster(current_path, con, max_cluster_size, min_cluster_size)


if __name__ == "__main__":
    max_cluster_size = 10
    min_cluster_size = 1

    con = duckdb.connect("data.db")
    con.execute("UPDATE abstracts SET path = NULL")

    con.execute("SELECT embeddings FROM abstracts;")
    results = con.fetchall()
    embeddings = [list(x[0]) for x in results]
    # con.execute("DROP TABLE tree;")

    con.execute(
        "CREATE TABLE IF NOT EXISTS tree (embeddings FLOAT[768],location INTEGER[])"
    )

    cluster(None, con)
