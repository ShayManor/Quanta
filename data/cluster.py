import time
from multiprocessing import Pool
from typing import List, Optional
import matplotlib.pyplot as plt
import duckdb
import numpy as np


def cluster_embeddings(embs):
    """
    Takes embeddings, clusters and returns the clusters
    :param embs: list of all available embeddings in latent space
    :return: list^2 of embeddings corresponding to clusters
    """
    return [embs[:len(embs) // 3], embs[len(embs) // 3:2 * len(embs) // 3], embs[2 * len(embs) // 3:]]


def get_avg(embs):
    """
    Gets average embedding of cluster for comparison
    :param embs: list of embeddings in the cluster - may be sampled
    :return: normalized embeddings
    """
    # if len(embs) > 100:
    #     embs = np.random.choice(embs, size=100, replace=False)

    return np.mean(embs, axis=0)[0].tolist()


def cluster(location: Optional[List[int]], db_path, max_cluster_size=10, min_cluster_size=1, level=0):
    """
    Recursively clusters embeddings into tree
    :param location: path (array of ints) to current location
    :param db_path: duckdb database path
    :param max_cluster_size: keeps splitting until less than max_cluster_size
    :param min_cluster_size: minimum size for cluster, > 1 could drop data.
    """
    con = duckdb.connect(db_path)
    if not location:
        con.execute("SELECT embeddings FROM abstracts order by id;")
        embeddings = con.fetchall()
    else:
        con.execute("SELECT embeddings FROM abstracts WHERE path = ? order by id;", [location])
        embeddings = con.fetchall()

    cluster_size = len(embeddings)

    print(location)
    print(f"Cluster size: {cluster_size}")
    if cluster_size < min_cluster_size:
        # Too small cluster - drop
        return
    if cluster_size <= max_cluster_size:
        # Finished cluster
        return

    clusters = cluster_embeddings(embeddings)
    for idx, current_cluster in enumerate(clusters):
        # Append index to path
        if not current_cluster:
            continue
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
        cluster(current_path, db_path, max_cluster_size, min_cluster_size, level + 1)
    con.commit()

def main(n_procs):
    start_time = time.time()
    max_cluster_size = 10
    min_cluster_size = 1

    con = duckdb.connect("data.db")
    con.execute("UPDATE abstracts SET path = NULL")

    con.execute(
        "CREATE TABLE IF NOT EXISTS tree (embeddings FLOAT[768],location INTEGER[])"
    )
    con.execute("DELETE FROM tree WHERE TRUE")
    with Pool(processes=n_procs) as pool:
        cluster(None, "data.db", level=0)
    print(f"Total time: {time.time() - start_time}")
    return time.time() - start_time

if __name__ == "__main__":
    n_procs_options = [1, 2, 3, 5, 10, 15, 20]
    res = []
    for n_proc in n_procs_options:
        res.append(main(n_proc))

    plt.plot(n_procs_options, res)
    plt.xlabel("Number of processes")
    plt.ylabel("Time to complete")

    plt.show()
