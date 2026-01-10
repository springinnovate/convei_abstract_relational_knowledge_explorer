import argparse
import sqlite3
import numpy as np


def load_embedding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fetch_one(cur, paper_id: int):
    cur.execute(
        """
        SELECT id, title, abstract, abstract_embedding
        FROM publications
        WHERE id = ?
        """,
        (paper_id,),
    )
    return cur.fetchone()


def iter_embeddings(cur):
    cur.execute(
        """
        SELECT id, title, abstract, abstract_embedding
        FROM publications
        WHERE abstract_embedding IS NOT NULL
        """
    )
    for row in cur:
        yield row


def knn_neighbors(conn, paper_id: int, k: int, min_sim: float):
    cur = conn.cursor()
    row = fetch_one(cur, paper_id)
    pid, title, abstract, emb_b = row
    e_p = normalize(load_embedding(emb_b))

    top_ids = np.empty(k, dtype=np.int64)
    top_sims = np.full(k, -np.inf, dtype=np.float32)
    top_titles = [None] * k
    top_abstracts = [None] * k
    top_embs = [None] * k

    for rid, rtitle, rabst, remb_b in iter_embeddings(cur):
        if rid == paper_id:
            continue
        e_r = load_embedding(remb_b)
        if e_r.shape != e_p.shape:
            continue
        e_rn = normalize(e_r)
        s = float(np.dot(e_p, e_rn))
        if s < min_sim:
            continue
        j = int(np.argmin(top_sims))
        if s > float(top_sims[j]):
            top_sims[j] = s
            top_ids[j] = rid
            top_titles[j] = rtitle
            top_abstracts[j] = rabst
            top_embs[j] = e_rn

    mask = np.isfinite(top_sims) & (top_sims > -np.inf)
    top_ids = top_ids[mask]
    top_sims = top_sims[mask]
    top_titles = [t for t, m in zip(top_titles, mask) if m]
    top_abstracts = [a for a, m in zip(top_abstracts, mask) if m]
    top_embs = [e for e, m in zip(top_embs, mask) if m]

    order = np.argsort(-top_sims)
    top_ids = top_ids[order]
    top_sims = top_sims[order]
    top_titles = [top_titles[i] for i in order]
    top_abstracts = [top_abstracts[i] for i in order]
    top_embs = [top_embs[i] for i in order]

    neighbors = []
    for rid, s, rtitle, rabst, e in zip(
        top_ids.tolist(), top_sims.tolist(), top_titles, top_abstracts, top_embs
    ):
        neighbors.append(
            {
                "id": rid,
                "sim_to_paper": float(s),
                "title": rtitle,
                "abstract": rabst,
                "emb": e,
            }
        )

    return {
        "paper": {
            "id": pid,
            "title": title,
            "abstract": abstract,
            "emb": e_p,
        },
        "neighbors": neighbors,
    }


def cluster_neighbors(embs: np.ndarray, min_cluster_size: int):
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric="euclidean"
        )
        labels = clusterer.fit_predict(embs)
        return labels
    except Exception:
        from sklearn.cluster import KMeans

        n = embs.shape[0]
        c = min(4, max(2, n // max(1, min_cluster_size)))
        labels = KMeans(
            n_clusters=c, n_init="auto", random_state=0
        ).fit_predict(embs)
        return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="2025_11_09_researchgate.sqlite")
    ap.add_argument("--paper-id", type=int, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--min-sim", type=float, default=0.3)
    ap.add_argument("--min-cluster-size", type=int, default=5)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    data = knn_neighbors(conn, args.paper_id, args.k, args.min_sim)

    paper = data["paper"]
    neighbors = data["neighbors"]

    print(f"* original paper")
    print(f'  * id: {paper["id"]}')
    print(f'  * title: {paper["title"]}')
    print(f'  * abstract: {paper["abstract"]}\n')

    if len(neighbors) == 0:
        return

    X = np.vstack([n["emb"] for n in neighbors]).astype(np.float32)
    labels = cluster_neighbors(X, args.min_cluster_size)

    clusters = {}
    for i, lab in enumerate(labels.tolist()):
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(i)

    items = []
    for lab, idxs in clusters.items():
        C = X[idxs]
        centroid = normalize(C.mean(axis=0))
        sim_p_centroid = float(np.dot(paper["emb"], centroid))
        best_i = None
        best_sim = -np.inf
        for i in idxs:
            s = float(np.dot(centroid, neighbors[i]["emb"]))
            if s > best_sim:
                best_sim = s
                best_i = i
        rep = neighbors[best_i]
        items.append((sim_p_centroid, lab, len(idxs), rep, best_sim))

    items.sort(key=lambda t: -t[0])

    print(
        f"* clusters (within min cosine similarity for neighbors: {args.min_sim})"
    )
    for sim_p_centroid, lab, size, rep, sim_rep_centroid in items:
        print(f"  * cluster {lab} (n={size})")
        print(
            f"    * cosine distance paper↔cluster: {1.0 - sim_p_centroid:.6f}"
        )
        print(
            f'    * representative (closest-to-centroid) paper id: {rep["id"]}'
        )
        print(f'    * representative title: {rep["title"]}')
        print(f'    * representative abstract: {rep["abstract"]}')
        print(
            f"    * cosine distance rep↔centroid: {1.0 - sim_rep_centroid:.6f}\n"
        )


if __name__ == "__main__":
    main()
