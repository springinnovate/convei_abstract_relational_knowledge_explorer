from __future__ import annotations

"""
Create publication geography figures from topic mapping report CSVs.

The script reads the newest CSVs matching the required report prefixes in an
input directory and writes high-resolution PNGs into a subdirectory. It uses:

- country_vs_year_*.csv for paper counts by country and year
- analyze_author_country_affiliation_type_weights_*.csv for all-author country
  signal, computed by summing affiliation type weights per country

Usage:
    python plot_publication_geography_figures.py --input-dir topic_mapping_reports
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import geodatasets
except ImportError:
    geodatasets = None


LOGGER = logging.getLogger(__name__)

COUNTRY_ALIASES = {
    "Cote Ivoire": "Cote d'Ivoire",
    "Peoples R China": "China",
    "People's R China": "China",
    "United States": "USA",
    "United States of America": "USA",
    "England": "UK",
    "Northern Ireland": "UK",
    "Scotland": "UK",
    "Wales": "UK",
    "United Kingdom": "UK",
    "Czechia": "Czech Republic",
}

COUNTRY_CENTROIDS = {
    "Afghanistan": (67.71, 33.94),
    "Albania": (20.17, 41.15),
    "Algeria": (1.66, 28.03),
    "Argentina": (-63.62, -38.42),
    "Australia": (133.78, -25.27),
    "Austria": (14.55, 47.52),
    "Bangladesh": (90.36, 23.68),
    "Belgium": (4.47, 50.50),
    "Brazil": (-51.93, -14.24),
    "Bulgaria": (25.49, 42.73),
    "Canada": (-106.35, 56.13),
    "Chile": (-71.54, -35.68),
    "China": (104.20, 35.86),
    "Colombia": (-74.30, 4.57),
    "Cote d'Ivoire": (-5.55, 7.54),
    "Croatia": (15.20, 45.10),
    "Czech Republic": (15.47, 49.82),
    "Denmark": (9.50, 56.26),
    "Egypt": (30.80, 26.82),
    "Estonia": (25.01, 58.60),
    "Finland": (25.75, 61.92),
    "France": (2.21, 46.23),
    "Germany": (10.45, 51.17),
    "Ghana": (-1.02, 7.95),
    "Greece": (21.82, 39.07),
    "Hungary": (19.50, 47.16),
    "India": (78.96, 20.59),
    "Indonesia": (113.92, -0.79),
    "Iran": (53.69, 32.43),
    "Iraq": (43.68, 33.22),
    "Ireland": (-8.24, 53.41),
    "Israel": (34.85, 31.05),
    "Italy": (12.57, 41.87),
    "Japan": (138.25, 36.20),
    "Jordan": (36.24, 30.59),
    "Kenya": (37.91, -0.02),
    "Malaysia": (101.98, 4.21),
    "Mexico": (-102.55, 23.63),
    "Morocco": (-7.09, 31.79),
    "Nepal": (84.12, 28.39),
    "Netherlands": (5.29, 52.13),
    "New Zealand": (174.89, -40.90),
    "Nigeria": (8.68, 9.08),
    "Norway": (8.47, 60.47),
    "Pakistan": (69.35, 30.38),
    "Peru": (-75.02, -9.19),
    "Philippines": (121.77, 12.88),
    "Poland": (19.15, 51.92),
    "Portugal": (-8.22, 39.40),
    "Romania": (24.97, 45.94),
    "Russia": (105.32, 61.52),
    "Saudi Arabia": (45.08, 23.89),
    "Singapore": (103.82, 1.35),
    "South Africa": (22.94, -30.56),
    "South Korea": (127.77, 35.91),
    "Spain": (-3.75, 40.46),
    "Sri Lanka": (80.77, 7.87),
    "Sweden": (18.64, 60.13),
    "Switzerland": (8.23, 46.82),
    "Taiwan": (120.96, 23.70),
    "Tanzania": (34.89, -6.37),
    "Thailand": (100.99, 15.87),
    "Tunisia": (9.54, 33.89),
    "Turkey": (35.24, 38.96),
    "UK": (-3.44, 55.38),
    "Ukraine": (31.17, 48.38),
    "United Arab Emirates": (53.85, 23.42),
    "USA": (-95.71, 37.09),
    "Vietnam": (108.28, 14.06),
}

COUNTRY_COLORS = [
    "#b74243",
    "#4e4aa8",
    "#f2a23b",
    "#62a85a",
    "#c7df45",
    "#65c96f",
    "#ef9aa6",
    "#7a7a7a",
    "#8c87d9",
    "#7cc9c6",
]


@dataclass
class FigureInputs:
    country_year_csv: Path
    author_country_affiliation_csv: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def latest_prefixed_csv(input_dir: Path, prefix: str, exclude_prefixes=()) -> Path:
    candidates = [
        path
        for path in input_dir.glob(f"{prefix}*.csv")
        if not any(path.name.startswith(excluded) for excluded in exclude_prefixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {input_dir} with prefix {prefix!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def discover_inputs(input_dir: Path) -> FigureInputs:
    return FigureInputs(
        country_year_csv=latest_prefixed_csv(
            input_dir,
            "country_vs_year_",
            exclude_prefixes=("country_vs_year_normalized_",),
        ),
        author_country_affiliation_csv=latest_prefixed_csv(
            input_dir,
            "analyze_author_country_affiliation_type_weights_",
        ),
    )


def clean_country_name(country: str) -> str:
    cleaned = str(country).strip()
    return COUNTRY_ALIASES.get(cleaned, cleaned)


def read_country_year(path: Path) -> pd.DataFrame:
    LOGGER.info("Reading country/year counts: %s", path)
    df = pd.read_csv(path)
    country_column = df.columns[0]
    df[country_column] = df[country_column].map(clean_country_name)
    year_columns = [column for column in df.columns[1:] if str(column).isdigit()]
    df[year_columns] = df[year_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    grouped = df.groupby(country_column, as_index=True)[year_columns].sum()
    grouped = grouped.reindex(sorted(grouped.index), axis=0)
    grouped.columns = [int(column) for column in grouped.columns]
    return grouped


def read_author_country_signal(path: Path) -> pd.Series:
    LOGGER.info("Reading author country affiliation weights: %s", path)
    df = pd.read_csv(path)
    country_column = df.columns[0]
    df[country_column] = df[country_column].map(clean_country_name)
    value_columns = df.columns[1:]
    df[value_columns] = df[value_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    signal = df.groupby(country_column)[value_columns].sum().sum(axis=1)
    return signal.sort_values(ascending=False)


def format_thousands(value, _position):
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def save_figure(fig, output_path: Path, dpi: int) -> None:
    LOGGER.info("Writing figure: %s", output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def top_countries(country_year: pd.DataFrame, top_n: int) -> list[str]:
    totals = country_year.sum(axis=1).sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def plot_annual_stacked_cumulative(
    country_year: pd.DataFrame,
    top_country_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    years = country_year.columns.tolist()
    annual_total = country_year.sum(axis=0)
    cumulative_total = annual_total.cumsum()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = np.zeros(len(years), dtype=float)

    for index, country in enumerate(top_country_names):
        values = country_year.loc[country, years].to_numpy(dtype=float)
        ax.bar(
            years,
            values,
            bottom=bottom,
            width=0.82,
            color=COUNTRY_COLORS[index % len(COUNTRY_COLORS)],
            edgecolor="white",
            linewidth=0.25,
            label=country,
        )
        bottom += values

    other_values = annual_total.to_numpy(dtype=float) - bottom
    ax.bar(
        years,
        other_values,
        bottom=bottom,
        width=0.82,
        color="#bdbdbd",
        edgecolor="white",
        linewidth=0.25,
        label="Other Countries",
    )

    cumulative_ax = ax.twinx()
    cumulative_ax.plot(
        years,
        cumulative_total,
        color="#333333",
        linewidth=1.8,
        linestyle=":",
        label="Total Cumulative Papers",
    )

    ax.set_title("Annual Papers by Country with Total Cumulative Papers", loc="left")
    ax.set_ylabel("Annual Papers per Country")
    cumulative_ax.set_ylabel("Total Cumulative Papers")
    ax.set_xlim(min(years) - 0.8, max(years) + 0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    cumulative_ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=55)

    handles, labels = ax.get_legend_handles_labels()
    line_handles, line_labels = cumulative_ax.get_legend_handles_labels()
    ax.legend(
        handles + line_handles,
        labels + line_labels,
        ncol=4,
        loc="upper left",
        frameon=False,
        fontsize=8.5,
    )

    save_figure(fig, output_path, dpi)


def plot_total_papers_bar(
    country_year: pd.DataFrame,
    top_country_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    totals = country_year.sum(axis=1).sort_values(ascending=False)
    top_totals = totals.loc[top_country_names]
    other_total = totals.drop(top_country_names).sum()
    plot_values = pd.concat([top_totals, pd.Series({"Other": other_total})])
    total_sum = plot_values.sum()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_positions = np.arange(len(plot_values))
    colors = COUNTRY_COLORS[: len(top_totals)] + ["#bdbdbd"]
    ax.barh(y_positions, plot_values.values, color=colors, edgecolor="white")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_values.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    ax.xaxis.tick_top()
    ax.set_title("Total Papers per Country", loc="left")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)

    for y_position, value in zip(y_positions, plot_values.values, strict=True):
        pct = value / total_sum * 100 if total_sum else 0.0
        ax.text(
            value,
            y_position,
            f" {value:,.0f} ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
        )

    save_figure(fig, output_path, dpi)


def load_world_geometries():
    if gpd is None:
        return None
    try:
        if geodatasets is not None:
            return gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception as exc:
        LOGGER.info("Could not load geodatasets Natural Earth land: %s", exc)
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception as exc:
        LOGGER.info("Could not load geopandas Natural Earth lowres: %s", exc)
    return None


def draw_world_background(ax) -> None:
    world = load_world_geometries()
    if world is not None:
        world.plot(
            ax=ax,
            color="#f3ead7",
            edgecolor="#6f6f6f",
            linewidth=0.35,
            zorder=1,
        )
    else:
        ax.set_facecolor("#faf7ef")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks(np.arange(-120, 181, 60))
    ax.set_yticks(np.arange(-60, 81, 20))
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def country_points(values: pd.Series) -> pd.DataFrame:
    rows = []
    missing = []
    for country, value in values.items():
        country_name = clean_country_name(country)
        centroid = COUNTRY_CENTROIDS.get(country_name)
        if centroid is None:
            missing.append(country_name)
            continue
        longitude, latitude = centroid
        rows.append(
            {
                "country": country_name,
                "longitude": longitude,
                "latitude": latitude,
                "value": float(value),
            }
        )
    if missing:
        LOGGER.warning(
            "No centroid for %d countries; first missing names: %s",
            len(missing),
            ", ".join(sorted(set(missing))[:12]),
        )
    return pd.DataFrame(rows)


def bubble_sizes(values: pd.Series, min_size: float, max_size: float) -> np.ndarray:
    if values.empty:
        return np.array([])
    scaled = np.sqrt(values / values.max())
    return min_size + scaled.to_numpy() * (max_size - min_size)


def plot_world_bubble_map(
    values: pd.Series,
    output_path: Path,
    dpi: int,
    title: str,
    color: str,
    legend_title: str,
) -> None:
    points = country_points(values)
    if points.empty:
        raise ValueError(f"No countries with known centroids for {title}")

    fig, ax = plt.subplots(figsize=(13, 7))
    draw_world_background(ax)
    sizes = bubble_sizes(points["value"], min_size=18, max_size=1500)
    ax.scatter(
        points["longitude"],
        points["latitude"],
        s=sizes,
        color=color,
        alpha=0.72,
        edgecolor="#4a3820",
        linewidth=0.65,
        zorder=3,
    )
    ax.set_title(title, loc="left")

    legend_values = np.geomspace(
        max(points["value"].min(), 1),
        points["value"].max(),
        num=5,
    )
    legend_sizes = bubble_sizes(pd.Series(legend_values), 18, 1500)
    handles = [
        ax.scatter([], [], s=size, color=color, alpha=0.72, edgecolor="#4a3820")
        for size in legend_sizes
    ]
    labels = [f"{value:,.0f}" for value in legend_values]
    ax.legend(
        handles,
        labels,
        title=legend_title,
        scatterpoints=1,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, -0.02),
        ncol=len(handles),
        fontsize=8,
        title_fontsize=9,
    )

    save_figure(fig, output_path, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("topic_mapping_reports"))
    parser.add_argument("--output-subdir", default="publication_geography_figures")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    setup_logging()

    input_dir = args.input_dir
    output_dir = input_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(input_dir)
    country_year = read_country_year(inputs.country_year_csv)
    author_signal = read_author_country_signal(inputs.author_country_affiliation_csv)
    top_country_names = top_countries(country_year, args.top_n)
    paper_totals = country_year.sum(axis=1).sort_values(ascending=False)

    LOGGER.info("Top countries: %s", ", ".join(top_country_names))

    plot_annual_stacked_cumulative(
        country_year,
        top_country_names,
        output_dir / "annual_papers_by_country_cumulative.png",
        args.dpi,
    )
    plot_total_papers_bar(
        country_year,
        top_country_names,
        output_dir / "total_papers_by_country.png",
        args.dpi,
    )
    plot_world_bubble_map(
        paper_totals,
        output_dir / "world_map_total_papers_by_country.png",
        args.dpi,
        "Total Papers by Country",
        "#a36b2c",
        "Papers",
    )
    plot_world_bubble_map(
        author_signal,
        output_dir / "world_map_author_country_signal.png",
        args.dpi,
        "Relative All-Author Country Signal",
        "#61bf7b",
        "Author signal",
    )

    LOGGER.info("Done. Figures written to %s", output_dir)


if __name__ == "__main__":
    main()
