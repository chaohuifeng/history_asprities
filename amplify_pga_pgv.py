"""
4. amplify_pga_pgv.py
Apply site amplification factors to simulated PGA/PGV results and generate interpolated
intensity maps for reporting. 场地放大与烈度插值脚本（对应 README 第 4 节）。
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, cKDTree

import math

try:
    import shapefile  # type: ignore
except ImportError:  # pragma: no cover
    shapefile = None

try:
    from shapely.geometry import shape as shapely_shape  # type: ignore
except Exception:  # pragma: no cover
    shapely_shape = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort:skip
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

FACTOR_TABLE = [
    {"class": "B", "vs30_min": 760, "vs30_max": float("inf"), "fa": 1.0, "fv": 1.0},
    {"class": "C", "vs30_min": 360, "vs30_max": 760, "fa": 1.2, "fv": 1.2},
    {"class": "D", "vs30_min": 180, "vs30_max": 360, "fa": 1.8, "fv": 1.5},
    {"class": "E", "vs30_min": 0, "vs30_max": 180, "fa": 2.0, "fv": 1.7},
]

DEFAULT_CLASS = "B"
DEFAULT_FA = 1.0
DEFAULT_FV = 1.0

SITE_EXTENSIONS = {".csv", ".parquet", ".feather", ".json", ".geojson", ".shp"}
EVENT_EXTENSIONS = {".csv", ".parquet", ".feather", ".json", ".geojson"}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amplify PGA/PGV using site factors.")
    parser.add_argument("--site_dir", required=True, help="Directory containing site files.")
    parser.add_argument("--gm_dir", required=True, help="Directory containing ground motion files.")
    parser.add_argument("--out_dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--match_km",
        type=float,
        default=2.0,
        help="Initial nearest-neighbour match threshold in kilometres (default: 2.0).",
    )
    return parser.parse_args()


def find_files(root: Path, allowed_ext: set) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_ext:
            files.append(path)
    return files


def centroid_from_shape(s) -> Optional[Tuple[float, float]]:
    if s is None:
        return None
    if shapely_shape is not None:
        try:
            geom = shapely_shape(s.__geo_interface__)
            point = geom.representative_point()
            return point.x, point.y
        except Exception:
            pass

    points = s.points
    if not points:
        return None

    parts = list(s.parts) + [len(points)]
    total_area = 0.0
    cx = 0.0
    cy = 0.0

    for idx in range(len(parts) - 1):
        ring = points[parts[idx]:parts[idx + 1]]
        if not ring:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        area = 0.0
        ring_cx = 0.0
        ring_cy = 0.0
        for i in range(len(ring) - 1):
            x0, y0 = ring[i]
            x1, y1 = ring[i + 1]
            cross = x0 * y1 - x1 * y0
            area += cross
            ring_cx += (x0 + x1) * cross
            ring_cy += (y0 + y1) * cross
        area *= 0.5
        if abs(area) < 1e-12:
            continue
        total_area += area
        cx += ring_cx
        cy += ring_cy

    if abs(total_area) < 1e-12:
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    cx /= (6.0 * total_area)
    cy /= (6.0 * total_area)
    return cx, cy


def normalize_site_class(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    if text in {"A", "F"}:
        return None
    if text in {"B", "C", "D", "E"}:
        return text
    if text.startswith("CLASS"):
        text = text.replace("CLASS", "").strip()
    if text in {"I", "II", "III", "IV"}:
        mapping = {"I": "B", "II": "C", "III": "D", "IV": "E"}
        return mapping.get(text)
    return None


def class_from_vs30(vs30: Optional[float]) -> str:
    if vs30 is None or not np.isfinite(vs30):
        return DEFAULT_CLASS
    for entry in FACTOR_TABLE:
        if entry["vs30_min"] <= vs30 < entry["vs30_max"]:
            return entry["class"]
    return DEFAULT_CLASS


def factors_for_class(site_class: Optional[str]) -> Tuple[str, float, float]:
    if site_class:
        site_class = site_class.upper()
    else:
        site_class = DEFAULT_CLASS
    for entry in FACTOR_TABLE:
        if entry["class"] == site_class:
            return site_class, entry["fa"], entry["fv"]
    return DEFAULT_CLASS, DEFAULT_FA, DEFAULT_FV


def identify_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for kw in keywords:
        if kw in lower_map:
            return lower_map[kw]
    for col in columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def load_generic_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in {".json", ".geojson"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_shapefile(path: Path) -> pd.DataFrame:
    if shapefile is None:
        raise ImportError("pyshp (shapefile) package is required to read shapefiles.")

    reader = shapefile.Reader(str(path))
    field_names = [f[0] for f in reader.fields[1:]]
    records = []
    vs30_missing = 0
    class_missing = 0

    for shape_record in reader.iterShapeRecords():
        centroid = centroid_from_shape(shape_record.shape)
        if centroid is None:
            continue
        lon, lat = centroid
        attr = {}
        for idx, field in enumerate(field_names):
            attr[field] = shape_record.record[idx]

        site_class = None
        vs30 = None
        for key, value in attr.items():
            lower_key = key.lower()
            if "vs" in lower_key and "30" in lower_key:
                try:
                    vs30 = float(value)
                except Exception:
                    vs30 = None
            elif "nehrp" in lower_key or ("site" in lower_key and "class" in lower_key):
                site_class = normalize_site_class(value)

        if site_class is None and vs30 is None:
            class_missing += 1
        if vs30 is None:
            vs30_missing += 1

        records.append(
            {
                "lon": lon,
                "lat": lat,
                "site_class": site_class,
                "vs30": vs30,
            }
        )

    log(f"Loaded shapefile {path.name}: {len(records)} site polygons.")
    if vs30_missing:
        log(f"  WARN: VS30 missing for {vs30_missing} polygons.")
    if class_missing:
        log(f"  INFO: Falling back to VS30-derived class for {class_missing} polygons.")
    return pd.DataFrame(records)


def load_site_tables(site_dir: Path) -> pd.DataFrame:
    files = find_files(site_dir, SITE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No site files found in {site_dir}.")

    frames = []
    for path in files:
        log(f"Reading site file: {path}")
        if path.suffix.lower() == ".shp":
            frame = load_shapefile(path)
        else:
            frame = load_generic_table(path)
        frames.append(frame)

    site_df = pd.concat(frames, ignore_index=True)

    lon_col = identify_column(site_df.columns.tolist(), ["lon", "longitude", "long", "x"])
    lat_col = identify_column(site_df.columns.tolist(), ["lat", "latitude", "y"])
    if lon_col is None or lat_col is None:
        if {"lon", "lat"}.issubset(site_df.columns):
            lon_col, lat_col = "lon", "lat"
        else:
            raise ValueError("Site data must include longitude and latitude columns.")

    site_df = site_df.rename(columns={lon_col: "lon", lat_col: "lat"})

    vs_col = identify_column(site_df.columns.tolist(), ["vs30", "vs_30", "v_s30"])
    class_col = identify_column(site_df.columns.tolist(), ["site_class", "siteclass", "nehrp"])

    if vs_col:
        site_df["vs30"] = pd.to_numeric(site_df[vs_col], errors="coerce")
    elif "vs30" in site_df.columns:
        site_df["vs30"] = pd.to_numeric(site_df["vs30"], errors="coerce")
    else:
        site_df["vs30"] = np.nan

    if class_col:
        site_df["site_class"] = site_df[class_col].apply(normalize_site_class)
    elif "site_class" not in site_df.columns:
        site_df["site_class"] = None

    site_df = site_df[["lon", "lat", "vs30", "site_class"]]
    site_df = site_df.dropna(subset=["lon", "lat"])

    missing_vs = site_df["vs30"].isna().sum()
    missing_class = site_df["site_class"].isna().sum()
    log(f"Site table assembled: {len(site_df)} points. Missing vs30: {missing_vs}, missing class: {missing_class}.")
    return site_df.reset_index(drop=True)


def build_kdtree(site_df: pd.DataFrame):
    radians = np.deg2rad(site_df["lat"].astype(float).to_numpy())
    cos_lat = np.cos(radians)
    x = site_df["lon"].astype(float).to_numpy() * cos_lat * 111.32
    y = site_df["lat"].astype(float).to_numpy() * 111.32
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    return tree


def detect_unit_adjustments(df: pd.DataFrame) -> Tuple[float, float, str]:
    max_pga = df["pga"].max()
    max_pgv = df["pgv"].max()
    factor_pga = 1.0
    factor_pgv = 1.0
    message = ""

    if max_pga <= 5 and max_pgv <= 0.5:
        factor_pga = 100.0
        factor_pgv = 100.0
        message = "Auto-converted PGA from m/s^2 to Gal and PGV from m/s to cm/s."
    elif max_pga <= 5:
        factor_pga = 100.0
        message = "Auto-converted PGA from m/s^2 to Gal."
    elif max_pgv <= 0.5:
        factor_pgv = 100.0
        message = "Auto-converted PGV from m/s to cm/s."

    return factor_pga, factor_pgv, message


def prepare_event_dataframe(path: Path) -> Tuple[str, pd.DataFrame]:
    name = path.stem
    df = load_generic_table(path)

    lon_col = identify_column(df.columns.tolist(), ["lon", "longitude", "long", "x"])
    lat_col = identify_column(df.columns.tolist(), ["lat", "latitude", "y"])
    pga_col = identify_column(df.columns.tolist(), ["pga", "pga (cm/s^2)", "pgac"])
    pgv_col = identify_column(df.columns.tolist(), ["pgv", "pgv (cm/s)", "pgvc"])

    if lon_col is None or lat_col is None or pga_col is None or pgv_col is None:
        raise ValueError(f"File {path} missing essential columns (lon, lat, pga, pgv).")

    df = df.rename(columns={lon_col: "lon", lat_col: "lat", pga_col: "pga", pgv_col: "pgv"})

    for col in ["lon", "lat", "pga", "pgv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lon", "lat", "pga", "pgv"])
    df = df[(df["pga"] > 0) & (df["pgv"] > 0)]
    removed = before_count - len(df)
    if removed:
        log(f"  INFO: Dropped {removed} non-finite or non-positive rows in {path.name}.")

    factor_pga, factor_pgv, unit_message = detect_unit_adjustments(df)
    if factor_pga != 1.0:
        df["pga"] *= factor_pga
    if factor_pgv != 1.0:
        df["pgv"] *= factor_pgv
    if unit_message:
        log(f"  WARN: {path.name} -> {unit_message}")

    return name, df.reset_index(drop=True)


def match_sites_to_events(
    event_df: pd.DataFrame,
    site_df: pd.DataFrame,
    tree: cKDTree,
    threshold_km: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    radians = np.deg2rad(event_df["lat"].to_numpy())
    cos_lat = np.cos(radians)
    event_x = event_df["lon"].to_numpy() * cos_lat * 111.32
    event_y = event_df["lat"].to_numpy() * 111.32
    event_coords = np.column_stack([event_x, event_y])

    distances, indices = tree.query(event_coords, k=1, distance_upper_bound=threshold_km)

    matched_classes: List[str] = []
    matched_fa: List[float] = []
    matched_fv: List[float] = []
    matched_vs30: List[Optional[float]] = []
    matched_distances: List[float] = []

    fallback_count = 0
    vs30_missing_count = 0

    for dist, idx in zip(distances, indices):
        if np.isinf(dist) or idx >= len(site_df):
            site_class = DEFAULT_CLASS
            fa = DEFAULT_FA
            fv = DEFAULT_FV
            fallback_count += 1
            matched_vs30.append(np.nan)
            matched_classes.append(site_class)
            matched_fa.append(fa)
            matched_fv.append(fv)
            matched_distances.append(float("nan"))
            continue

        site_row = site_df.iloc[int(idx)]
        site_class = site_row.get("site_class")
        vs30 = site_row.get("vs30")
        if pd.isna(vs30):
            vs30 = None
        else:
            vs30 = float(vs30)

        if not site_class:
            site_class = class_from_vs30(vs30)
            if vs30 is None:
                vs30_missing_count += 1
        site_class, fa, fv = factors_for_class(site_class)

        matched_vs30.append(vs30 if vs30 is not None else np.nan)
        matched_classes.append(site_class)
        matched_fa.append(fa)
        matched_fv.append(fv)
        matched_distances.append(float(dist))

    matched = event_df.copy()
    matched["site_class"] = matched_classes
    matched["vs30"] = matched_vs30
    matched["Fa"] = matched_fa
    matched["Fv"] = matched_fv
    matched["match_distance_km"] = matched_distances

    stats = {
        "fallback_default": fallback_count,
        "missing_vs30_used_class_from_default": vs30_missing_count,
    }
    return matched, stats


def compute_intensity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PGA_original"] = df["pga"]
    df["PGV_original"] = df["pgv"]
    df["PGA_amplified"] = df["pga"] * df["Fa"]
    df["PGV_amplified"] = df["pgv"] * df["Fv"]

    pga_m = df["PGA_amplified"] / 100.0
    pgv_m = df["PGV_amplified"] / 100.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ipga = 3.17 * np.log10(pga_m) + 6.59
        ipgv = 3.17 * np.log10(pgv_m) + 9.77

    ipga = np.where(pga_m > 0, ipga, np.nan)
    ipgv = np.where(pgv_m > 0, ipgv, np.nan)

    df["IPGA"] = np.clip(ipga, None, 11.0)
    df["IPGV"] = np.clip(ipgv, None, 11.0)

    cond = (df["IPGA"] >= 6.0) & (df["IPGV"] >= 6.0)
    df["Intensity"] = np.where(cond, df["IPGV"], 0.5 * (df["IPGA"] + df["IPGV"]))
    df["Intensity"] = np.clip(df["Intensity"], None, 11.0)
    return df


def summarize_event(name: str, df: pd.DataFrame) -> str:
    summary = []
    summary.append(f"Event: {name}")
    summary.append(f"  Samples: {len(df)}")
    summary.append(f"  PGA mean (orig/amp): {df['PGA_original'].mean():.3f} / {df['PGA_amplified'].mean():.3f}")
    summary.append(f"  PGA max (orig/amp): {df['PGA_original'].max():.3f} / {df['PGA_amplified'].max():.3f}")
    summary.append(f"  PGV mean (orig/amp): {df['PGV_original'].mean():.3f} / {df['PGV_amplified'].mean():.3f}")
    summary.append(f"  PGV max (orig/amp): {df['PGV_original'].max():.3f} / {df['PGV_amplified'].max():.3f}")
    summary.append(f"  Intensity p50/p90: {df['Intensity'].quantile(0.5):.3f} / {df['Intensity'].quantile(0.9):.3f}")
    return "\n".join(summary)


def plot_event_map(df: pd.DataFrame, out_dir: Path, event_name: str) -> None:
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    inten = df["Intensity"].to_numpy()

    if len(df) < 3:
        log("  WARN: Not enough points for interpolation; falling back to scatter plot.")
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(lon, lat, c=inten, cmap="plasma", s=30, edgecolor="k", linewidths=0.2)
        plt.colorbar(sc, label="Intensity")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{event_name} Intensity (Amplified)")
        plt.tight_layout()
        out_path = out_dir / f"{event_name}_intensity.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        log(f"  Plot saved: {out_path}")
        return

    grid_lon = np.linspace(lon.min(), lon.max(), 200)
    grid_lat = np.linspace(lat.min(), lat.max(), 200)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    grid_z = griddata((lon, lat), inten, (grid_x, grid_y), method="cubic")
    if np.all(np.isnan(grid_z)):
        grid_z = griddata((lon, lat), inten, (grid_x, grid_y), method="linear")

    if len(df) >= 3:
        try:
            hull = ConvexHull(np.column_stack([lon, lat]))
            hull_points = np.column_stack([lon, lat])[hull.vertices]
            hull_path = MplPath(hull_points)
            mask = hull_path.contains_points(np.column_stack([grid_x.flatten(), grid_y.flatten()]))
            mask = mask.reshape(grid_x.shape)
            grid_z = np.where(mask, grid_z, np.nan)
        except Exception:
            pass

    if np.all(np.isnan(grid_z)):
        log("  WARN: Interpolation produced NaNs only; falling back to scatter plot.")
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(lon, lat, c=inten, cmap="plasma", s=30, edgecolor="k", linewidths=0.2)
        plt.colorbar(sc, label="Intensity")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{event_name} Intensity (Amplified)")
        plt.tight_layout()
        out_path = out_dir / f"{event_name}_intensity.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        log(f"  Plot saved: {out_path}")
        return

    valid = np.isfinite(grid_z)
    if valid.sum() == 0:
        log("  WARN: No valid interpolated values; skipping contour plot.")
        return

    data_min = float(np.nanmin(grid_z))
    data_max = float(np.nanmax(grid_z))
    lower = max(4.5, np.floor(data_min * 2.0) / 2.0)
    upper = min(11.5, np.ceil(data_max * 2.0) / 2.0)
    if lower >= upper:
        lower = max(4.5, data_min - 0.5)
        upper = min(11.5, data_max + 0.5)

    bounds = np.arange(lower, upper + 0.5, 0.5)
    if len(bounds) < 3:
        bounds = np.linspace(lower, upper, 3)

    cmap = plt.get_cmap("Spectral_r", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(grid_x, grid_y, grid_z, levels=bounds, cmap=cmap, norm=norm, alpha=0.85)
    contour_lines = plt.contour(grid_x, grid_y, grid_z, levels=np.arange(np.floor(lower), np.ceil(upper)), colors="k", linewidths=0.6)

    level_map = {5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI"}

    def roman_formatter(value: float) -> str:
        rounded = int(round(value))
        return level_map.get(rounded, f"{value:.1f}")

    try:
        plt.clabel(contour_lines, fmt=lambda v: roman_formatter(v), fontsize=9, inline=True)
    except Exception:
        pass

    plt.scatter(lon, lat, c="k", s=10, alpha=0.5, label="Stations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{event_name} Intensity (Amplified)")
    cbar = plt.colorbar(contourf)
    cbar.set_label("Intensity (MSK)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = out_dir / f"{event_name}_intensity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"  Plot saved: {out_path}")


def run_pipeline(site_dir: Path, gm_dir: Path, out_dir: Path, threshold_km: float) -> Tuple[int, int]:
    if not site_dir.is_dir():
        raise NotADirectoryError(f"Site directory not found: {site_dir}")
    if not gm_dir.is_dir():
        raise NotADirectoryError(f"Ground motion directory not found: {gm_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    site_df = load_site_tables(site_dir)
    tree = build_kdtree(site_df)

    gm_files = find_files(gm_dir, EVENT_EXTENSIONS)
    if len(gm_files) < 8:
        log(f"  WARN: Expected at least 8 event files, found {len(gm_files)}.")

    results = []
    summaries = []
    fallback_total = 0
    sample_total = 0

    for path in sorted(gm_files):
        log(f"Processing event file: {path.name}")
        event_name, event_df = prepare_event_dataframe(path)
        if event_df.empty:
            log(f"  WARN: Event {event_name} has no valid data after filtering. Skipping.")
            continue

        matched_df, match_stats = match_sites_to_events(event_df, site_df, tree, threshold_km)
        default_count = match_stats["fallback_default"]
        missing_vs30_count = match_stats["missing_vs30_used_class_from_default"]
        if default_count:
            log(f"  WARN: {default_count} points fell outside {threshold_km} km; defaulted to class B.")
        if missing_vs30_count:
            log(f"  INFO: {missing_vs30_count} points lacked VS30; class from defaults.")

        enriched = compute_intensity(matched_df)
        enriched.insert(0, "event", event_name)
        summary = summarize_event(event_name, enriched)
        summaries.append(summary)
        log(summary)

        plot_event_map(enriched, out_dir, event_name)

        out_path = out_dir / f"{event_name}_amplified.csv"
        enriched.to_csv(out_path, index=False)
        log(f"  Saved: {out_path}")

        results.append(enriched)
        fallback_total += default_count
        sample_total += len(enriched)

    if not results:
        raise RuntimeError("No event data processed successfully.")

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(out_dir / "all_events_amplified.csv", index=False)

    summary_txt = "\n\n".join(summaries)
    (out_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")
    log(f"Summary saved: {out_dir / 'summary.txt'}")
    return fallback_total, sample_total


def main() -> None:
    args = parse_args()
    site_dir = Path(args.site_dir).resolve()
    gm_dir = Path(args.gm_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    log(f"Starting amplification pipeline with threshold {args.match_km:.2f} km.")
    try:
        fallback_total, sample_total = run_pipeline(site_dir, gm_dir, out_dir, threshold_km=args.match_km)
    except Exception as exc:
        log(f"ERROR: {exc}")
        log("Attempting self-heal by retrying with 5.0 km threshold.")
        fallback_total, sample_total = run_pipeline(site_dir, gm_dir, out_dir, threshold_km=5.0)
    else:
        if sample_total > 0:
            fallback_ratio = fallback_total / sample_total
            if fallback_ratio > 0.5 and args.match_km < 5.0:
                log(
                    f"High fallback ratio detected ({fallback_ratio:.1%}); "
                    "retrying automatically with 5.0 km threshold."
                )
                fallback_total, sample_total = run_pipeline(site_dir, gm_dir, out_dir, threshold_km=5.0)
                if sample_total > 0:
                    log(f"Final fallback ratio: {fallback_total / sample_total:.1%}")

    log("DONE")
    log(f"Outputs available in: {out_dir}")


if __name__ == "__main__":
    main()

