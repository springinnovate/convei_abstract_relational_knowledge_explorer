import argparse
import csv
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import MetaData, Table, create_engine, event, select
from sqlalchemy.orm import sessionmaker


_SPLIT_RE = re.compile(r"[;,]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", required=True)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--top-pairs", type=int, default=50)
    p.add_argument("--top-pairs-per-year", type=int, default=20)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--include-unknown", action="store_true", default=False)
    return p.parse_args()


def _norm_token(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s


def parse_satellite_list(raw: str | None, include_unknown: bool) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    tokens = [_norm_token(t) for t in _SPLIT_RE.split(raw)]
    tokens = [t for t in tokens if t]
    if not include_unknown:
        drop = {
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "unspecified",
            "not specified",
        }
        tokens = [t for t in tokens if t.casefold() not in drop]
    return tokens


@dataclass
class PubRow:
    pub_id: int
    year: int
    sats: list[str]


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_top_satellites_lines(
    out_path: str,
    years: list[int],
    sats: list[str],
    counts_by_year: dict[int, Counter],
) -> None:
    plt.figure(figsize=(14, 7))
    for sat in sats:
        y = [counts_by_year.get(yr, Counter()).get(sat, 0) for yr in years]
        plt.plot(years, y, linewidth=2, label=sat)
    plt.title("Top satellites by year (publication mentions)")
    plt.xlabel("Year")
    plt.ylabel("Count (publications mentioning satellite)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_satellites_heatmap(
    out_path: str,
    years: list[int],
    sats: list[str],
    counts_by_year: dict[int, Counter],
) -> None:
    mat = np.zeros((len(sats), len(years)), dtype=float)
    for i, sat in enumerate(sats):
        for j, yr in enumerate(years):
            mat[i, j] = counts_by_year.get(yr, Counter()).get(sat, 0)

    plt.figure(figsize=(14, max(6, 0.35 * len(sats))))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.02, pad=0.02, label="Count")
    plt.yticks(range(len(sats)), sats, fontsize=9)
    xt = list(range(0, len(years), max(1, len(years) // 15)))
    plt.xticks(xt, [str(years[i]) for i in xt], rotation=45, ha="right")
    plt.title("Top satellites heatmap (mentions per year)")
    plt.xlabel("Year")
    plt.ylabel("Satellite")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    engine = create_engine(
        f"sqlite:///{args.db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    Session = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    md = MetaData()
    pubs = Table("publications", md, autoload_with=engine)

    ensure_dir(args.out_dir)

    YEAR_MIN = 1970
    YEAR_MAX = 2025

    with Session() as session:
        q = select(
            pubs.c.id,
            pubs.c.publication_year,
            pubs.c.satellite_type,
        ).where(
            pubs.c.publication_year.is_not(None),
            pubs.c.publication_year >= YEAR_MIN,
            pubs.c.publication_year <= YEAR_MAX,
        )

        if args.min_year is not None:
            q = q.where(pubs.c.publication_year >= max(args.min_year, YEAR_MIN))
        if args.max_year is not None:
            q = q.where(pubs.c.publication_year <= min(args.max_year, YEAR_MAX))

        rows = session.execute(q).all()

    pub_rows: list[PubRow] = []
    for pub_id, year, sat_raw in rows:
        try:
            year_i = int(year)
        except Exception:
            continue
        sats = parse_satellite_list(sat_raw, args.include_unknown)
        if not sats:
            continue
        pub_rows.append(PubRow(pub_id=int(pub_id), year=year_i, sats=sats))

    if not pub_rows:
        write_csv(
            os.path.join(args.out_dir, "README.csv"),
            ["message"],
            [
                [
                    "No rows with (publication_year, satellite_type) found after filtering."
                ]
            ],
        )
        return

    years = sorted({r.year for r in pub_rows})
    year_min, year_max = years[0], years[-1]
    years = list(range(year_min, year_max + 1))

    overall = Counter()
    sat_first_year: dict[str, int] = {}
    sat_last_year: dict[str, int] = {}
    counts_by_year: dict[int, Counter] = defaultdict(Counter)
    pubs_by_year: dict[int, set[int]] = defaultdict(set)
    sats_per_pub_by_year: dict[int, list[int]] = defaultdict(list)

    for r in pub_rows:
        pubs_by_year[r.year].add(r.pub_id)
        unique_sats = []
        seen = set()
        for s in r.sats:
            if s in seen:
                continue
            seen.add(s)
            unique_sats.append(s)

        sats_per_pub_by_year[r.year].append(len(unique_sats))

        for sat in unique_sats:
            overall[sat] += 1
            counts_by_year[r.year][sat] += 1
            if sat not in sat_first_year or r.year < sat_first_year[sat]:
                sat_first_year[sat] = r.year
            if sat not in sat_last_year or r.year > sat_last_year[sat]:
                sat_last_year[sat] = r.year

    overall_rows = []
    for sat, cnt in overall.most_common():
        if cnt < args.min_count:
            continue
        overall_rows.append(
            [sat, cnt, sat_first_year.get(sat, ""), sat_last_year.get(sat, "")]
        )
    write_csv(
        os.path.join(args.out_dir, "satellites_overall.csv"),
        ["satellite", "count_publications", "first_year", "last_year"],
        overall_rows,
    )

    top_sats = [
        sat for sat, cnt in overall.most_common() if cnt >= args.min_count
    ][: args.top_n]

    by_year_header = [
        "year",
        "total_pubs_with_satellite_type",
        "unique_satellites_that_year",
        "total_mentions_that_year",
    ] + top_sats
    by_year_rows = []
    for yr in years:
        c = counts_by_year.get(yr, Counter())
        total_pubs = len(pubs_by_year.get(yr, set()))
        uniq = len(c)
        total_mentions = sum(c.values())
        row = [yr, total_pubs, uniq, total_mentions] + [
            c.get(s, 0) for s in top_sats
        ]
        by_year_rows.append(row)
    write_csv(
        os.path.join(args.out_dir, "satellites_by_year_top.csv"),
        by_year_header,
        by_year_rows,
    )

    year_summary_rows = []
    for yr in years:
        sizes = sats_per_pub_by_year.get(yr, [])
        if sizes:
            mean_v = sum(sizes) / len(sizes)
            med_v = statistics.median(sizes)
        else:
            mean_v = 0.0
            med_v = 0.0
        year_summary_rows.append(
            [
                yr,
                len(pubs_by_year.get(yr, set())),
                len(counts_by_year.get(yr, Counter())),
                round(mean_v, 3),
                round(med_v, 3),
            ]
        )
    write_csv(
        os.path.join(args.out_dir, "year_summary.csv"),
        [
            "year",
            "publications_with_satellite_type",
            "unique_satellites",
            "mean_satellites_per_publication",
            "median_satellites_per_publication",
        ],
        year_summary_rows,
    )

    top_per_year_rows = []
    for yr in years:
        c = counts_by_year.get(yr, Counter())
        for rank, (sat, cnt) in enumerate(c.most_common(25), start=1):
            top_per_year_rows.append([yr, rank, sat, cnt])

    write_csv(
        os.path.join(args.out_dir, "top_satellites_per_year.csv"),
        ["year", "rank", "satellite", "count_publications"],
        top_per_year_rows,
    )

    pair_overall = Counter()
    pair_by_year: dict[int, Counter] = defaultdict(Counter)
    for r in pub_rows:
        uniq = sorted(set(r.sats))
        for a, b in combinations(uniq, 2):
            pair_overall[(a, b)] += 1
            pair_by_year[r.year][(a, b)] += 1

    pair_overall_rows = []
    for (a, b), cnt in pair_overall.most_common(args.top_pairs):
        pair_overall_rows.append([a, b, cnt])
    write_csv(
        os.path.join(args.out_dir, "cooccurrence_pairs_overall.csv"),
        ["satellite_a", "satellite_b", "count_publications"],
        pair_overall_rows,
    )

    pair_by_year_rows = []
    for yr in years:
        c = pair_by_year.get(yr, Counter())
        for rank, ((a, b), cnt) in enumerate(
            c.most_common(args.top_pairs_per_year), start=1
        ):
            pair_by_year_rows.append([yr, rank, a, b, cnt])
    write_csv(
        os.path.join(args.out_dir, "cooccurrence_pairs_by_year.csv"),
        ["year", "rank", "satellite_a", "satellite_b", "count_publications"],
        pair_by_year_rows,
    )

    plot_top_satellites_lines(
        os.path.join(args.out_dir, "plot_top_satellites_lines.png"),
        years,
        top_sats,
        counts_by_year,
    )
    plot_top_satellites_heatmap(
        os.path.join(args.out_dir, "plot_top_satellites_heatmap.png"),
        years,
        top_sats,
        counts_by_year,
    )


if __name__ == "__main__":
    main()
