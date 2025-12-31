import os
import json
import math
import sqlite3
from array import array

import numpy as np
import faiss
import igraph as ig
import leidenalg
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SQLITE_PATH = "2025_11_09_researchgate.sqlite"
OUT_DIR = "topic_out"

K = 30
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 80

LEIDEN_RESOLUTION = 1.0
LEIDEN_N_ITER = -1

EMB_MEMMAP_PATH = os.path.join(OUT_DIR, "embeddings.f32.mmap")
ID_MEMMAP_PATH = os.path.join(OUT_DIR, "pub_ids.i64.mmap")

EDGE_CHUNK = 250_000
ADD_BATCH = 50_000
SEARCH_BATCH = 20_000

TOP_TOPICS = 99999
REPS_PER_TOPIC = 5
KEYWORD_SAMPLE = 2500
KEYWORDS_PER_TOPIC = 15

np.random.seed(7)


def _sql_in_chunks(cur, base_sql, ids, chunk=900):
    out = []
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        q = base_sql.format(",".join(["?"] * len(part)))
        cur.execute(q, part)
        out.extend(cur.fetchall())
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    con = sqlite3.connect(SQLITE_PATH)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    cur = con.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM publications WHERE abstract_embedding IS NOT NULL"
    )
    n = int(cur.fetchone()[0])

    cur.execute(
        "SELECT abstract_embedding FROM publications WHERE abstract_embedding IS NOT NULL LIMIT 1"
    )
    first_blob = cur.fetchone()[0]
    dim = len(first_blob) // 4

    X = np.memmap(EMB_MEMMAP_PATH, dtype="float32", mode="w+", shape=(n, dim))
    pub_ids = np.memmap(ID_MEMMAP_PATH, dtype="int64", mode="w+", shape=(n,))

    cur.execute(
        "SELECT id, abstract_embedding FROM publications WHERE abstract_embedding IS NOT NULL ORDER BY id"
    )

    i = 0
    pbar = tqdm(total=n, desc="Loading + normalizing embeddings")
    while True:
        rows = cur.fetchmany(ADD_BATCH)
        if not rows:
            break
        m = len(rows)
        ids = np.fromiter((r[0] for r in rows), dtype="int64", count=m)
        pub_ids[i : i + m] = ids
        for j, (_, blob) in enumerate(rows):
            X[i + j] = np.frombuffer(blob, dtype=np.float32, count=dim)
        norms = np.linalg.norm(X[i : i + m], axis=1, keepdims=True)
        X[i : i + m] /= norms
        i += m
        pbar.update(m)
    pbar.close()
    X.flush()
    pub_ids.flush()

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH

    pbar = tqdm(total=n, desc="Building FAISS HNSW index")
    for i in range(0, n, ADD_BATCH):
        xb = np.asarray(X[i : i + ADD_BATCH])
        index.add(xb)
        pbar.update(xb.shape[0])
    pbar.close()

    g = ig.Graph(n=n, directed=False)
    edge_buf = []
    weight_buf = []

    pbar = tqdm(total=n, desc="kNN search + graph build")
    for i0 in range(0, n, SEARCH_BATCH):
        xb = np.asarray(X[i0 : i0 + SEARCH_BATCH])
        D, I = index.search(xb, K + 1)
        I = I[:, 1:]
        D = D[:, 1:]
        for r in range(I.shape[0]):
            src = i0 + r
            nbrs = I[r]
            sims = D[r]
            for j in range(K):
                dst = int(nbrs[j])
                if src < dst:
                    edge_buf.append((src, dst))
                    weight_buf.append(float(sims[j]))
            if len(edge_buf) >= EDGE_CHUNK:
                e0 = g.ecount()
                g.add_edges(edge_buf)
                g.es[e0 : g.ecount()]["weight"] = weight_buf
                edge_buf.clear()
                weight_buf.clear()
        pbar.update(xb.shape[0])
    pbar.close()

    if edge_buf:
        e0 = g.ecount()
        g.add_edges(edge_buf)
        g.es[e0 : g.ecount()]["weight"] = weight_buf
        edge_buf.clear()
        weight_buf.clear()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=LEIDEN_RESOLUTION,
        n_iterations=LEIDEN_N_ITER,
    )

    membership = np.asarray(partition.membership, dtype="int32")
    n_topics = int(membership.max()) + 1
    sizes = np.bincount(membership, minlength=n_topics).astype("int64")

    with open(
        os.path.join(OUT_DIR, "publication_topics.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("publication_id,topic_id\n")
        for i in range(n):
            f.write(f"{int(pub_ids[i])},{int(membership[i])}\n")

    centroids = np.zeros((n_topics, dim), dtype="float64")
    pbar = tqdm(total=n, desc="Computing topic centroids")
    for i0 in range(0, n, ADD_BATCH):
        xb = np.asarray(X[i0 : i0 + ADD_BATCH])
        tb = membership[i0 : i0 + xb.shape[0]]
        for t in np.unique(tb):
            mask = tb == t
            centroids[int(t)] += xb[mask].sum(axis=0)
        pbar.update(xb.shape[0])
    pbar.close()

    centroids /= np.maximum(sizes[:, None], 1)
    cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = (centroids / cent_norms).astype("float32")

    best_scores = np.full((n_topics, REPS_PER_TOPIC), -1.0, dtype="float32")
    best_rows = np.full((n_topics, REPS_PER_TOPIC), -1, dtype="int64")

    pbar = tqdm(total=n, desc="Finding representative papers")
    for i0 in range(0, n, ADD_BATCH):
        xb = np.asarray(X[i0 : i0 + ADD_BATCH])
        tb = membership[i0 : i0 + xb.shape[0]]
        cb = centroids[tb]
        scores = (xb * cb).sum(axis=1).astype("float32")
        for r in range(xb.shape[0]):
            t = int(tb[r])
            s = float(scores[r])
            br = best_rows[t]
            bs = best_scores[t]
            k = int(np.argmin(bs))
            if s > float(bs[k]):
                bs[k] = s
                br[k] = i0 + r
        pbar.update(xb.shape[0])
    pbar.close()

    top_topics = np.argsort(-sizes)[:TOP_TOPICS].astype("int32").tolist()

    topic_summaries = []
    cur2 = con.cursor()

    for t in tqdm(top_topics, desc="Summarizing top topics"):
        rows = best_rows[t]
        rows = rows[rows >= 0].tolist()
        rep_pub_ids = [int(pub_ids[r]) for r in rows]

        rep_meta = _sql_in_chunks(
            cur2,
            "SELECT id, title, doi, publication_year FROM publications WHERE id IN ({})",
            rep_pub_ids,
            chunk=900,
        )
        rep_meta_map = {int(r[0]): (r[1], r[2], r[3]) for r in rep_meta}
        reps = []
        for pid in rep_pub_ids:
            title, doi, year = rep_meta_map.get(pid, ("", None, None))
            reps.append({"id": pid, "year": year, "doi": doi, "title": title})

        idxs = np.where(membership == t)[0]
        if idxs.size > KEYWORD_SAMPLE:
            idxs = np.random.choice(idxs, size=KEYWORD_SAMPLE, replace=False)
        sample_pub_ids = [int(pub_ids[i]) for i in idxs.tolist()]

        abs_rows = _sql_in_chunks(
            cur2,
            "SELECT abstract FROM publications WHERE id IN ({}) AND abstract IS NOT NULL",
            sample_pub_ids,
            chunk=900,
        )
        texts = [r[0] for r in abs_rows if r[0]]
        keywords = []
        if texts:
            vec = TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
            )
            Xtf = vec.fit_transform(texts)
            scores = np.asarray(Xtf.mean(axis=0)).ravel()
            terms = np.asarray(vec.get_feature_names_out())
            topk = np.argsort(-scores)[:KEYWORDS_PER_TOPIC]
            keywords = terms[topk].tolist()

        topic_summaries.append(
            {
                "topic_id": int(t),
                "size": int(sizes[t]),
                "keywords": keywords,
                "representatives": reps,
            }
        )

    with open(os.path.join(OUT_DIR, "topics.jsonl"), "w", encoding="utf-8") as f:
        for obj in topic_summaries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    nn = NearestNeighbors(
        n_neighbors=min(11, n_topics), metric="cosine", algorithm="auto"
    )
    nn.fit(centroids)
    dists, nbrs = nn.kneighbors(centroids)

    with open(os.path.join(OUT_DIR, "topic_edges.csv"), "w", encoding="utf-8") as f:
        f.write("topic_a,topic_b,similarity\n")
        for a in range(n_topics):
            for j in range(1, nbrs.shape[1]):
                b = int(nbrs[a, j])
                sim = 1.0 - float(dists[a, j])
                if a < b:
                    f.write(f"{a},{b},{sim}\n")

    print(f"n_pubs={n}")
    print(f"dim={dim}")
    print(f"n_topics={n_topics}")
    print(
        "top_topics_by_size="
        + ",".join(str(int(t)) for t in top_topics[: min(10, len(top_topics))])
    )
    print(f"outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
