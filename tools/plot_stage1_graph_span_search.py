#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RRAA_ORDER = ["1,2,3", "1,2,3,5", "1,2,3,5,8", "1,2,3,5,8,13"]
TRACKS_ORDER = ["1", "1,2,3", "1,2,3,5"]


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def best_per_protocol(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["deltas_tracks", "deltas_rraa"]
    idx = df.groupby(cols)["score"].idxmin()
    return df.loc[idx].sort_values("score").reset_index(drop=True)


def best_per_rraa(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("deltas_rraa")["score"].idxmin()
    out = df.loc[idx].copy()
    out["deltas_rraa"] = pd.Categorical(out["deltas_rraa"], RRAA_ORDER, ordered=True)
    return out.sort_values("deltas_rraa").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Stage-1 graph-span search results.")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Stage 1 graph-span search")
    ap.add_argument("--window", type=int, default=3, help="Rolling window for local mean/RMS curves.")
    args = ap.parse_args()
    window = max(1, int(args.window))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.trials_csv)
    df = df[df["state"].astype(str).str.endswith("COMPLETE")].copy()
    df = to_num(
        df,
        [
            "trial_number",
            "score",
            "mean_primary_metric",
            "std_primary_metric",
            "worst_primary_metric",
            "mean_rotation_median_deg",
            "failure_rate",
        ],
    )
    df = df.sort_values("trial_number").reset_index(drop=True)
    df["best_so_far"] = df["score"].cummin()
    df["rolling_mean"] = df["score"].rolling(window=window, min_periods=1).mean()
    df["rolling_rms"] = np.sqrt((df["score"] ** 2).rolling(window=window, min_periods=1).mean())
    unique = best_per_protocol(df)
    unique.to_csv(out_dir / "stage1_graph_span_unique_protocols.csv", index=False)

    best = df.loc[df["score"].idxmin()]
    best_rraa = str(best["deltas_rraa"])
    best_tracks = str(best["deltas_tracks"])

    fig = plt.figure(figsize=(14.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.35])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(df["trial_number"], df["score"], s=34, color="#9aa3ad", alpha=0.75, label="trial")
    ax0.plot(
        df["trial_number"],
        df["rolling_mean"],
        color="#2563eb",
        lw=1.8,
        alpha=0.9,
        label=f"{window}-trial mean",
    )
    ax0.plot(
        df["trial_number"],
        df["rolling_rms"],
        color="#16a34a",
        lw=1.8,
        alpha=0.9,
        linestyle="--",
        label=f"{window}-trial RMS",
    )
    ax0.plot(df["trial_number"], df["best_so_far"], color="#b42318", lw=2.2, label="best so far")
    ax0.scatter([best["trial_number"]], [best["score"]], s=80, color="#b42318", zorder=5)
    ax0.annotate(
        f"best: tracks {best_tracks}\nRRAA {best_rraa}",
        xy=(best["trial_number"], best["score"]),
        xytext=(0.42, 0.28),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"),
        fontsize=9,
    )
    ax0.set_yscale("log")
    ax0.set_title("Optimization history")
    ax0.set_xlabel("Trial")
    ax0.set_ylabel("Search score (log, lower is better)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, loc="upper right")

    heat = np.full((len(TRACKS_ORDER), len(RRAA_ORDER)), np.nan)
    for _, row in unique.iterrows():
        if row["deltas_tracks"] in TRACKS_ORDER and row["deltas_rraa"] in RRAA_ORDER:
            yi = TRACKS_ORDER.index(row["deltas_tracks"])
            xi = RRAA_ORDER.index(row["deltas_rraa"])
            heat[yi, xi] = row["score"]

    ax1 = fig.add_subplot(gs[0, 1])
    finite = heat[np.isfinite(heat)]
    im = ax1.imshow(
        heat,
        cmap="viridis_r",
        vmin=float(np.nanmin(finite)),
        vmax=float(np.nanpercentile(finite, 85)),
        aspect="auto",
    )
    ax1.set_title("Best score by graph span")
    ax1.set_xticks(range(len(RRAA_ORDER)), RRAA_ORDER, rotation=35, ha="right")
    ax1.set_yticks(range(len(TRACKS_ORDER)), TRACKS_ORDER)
    ax1.set_xlabel("RRAA deltas")
    ax1.set_ylabel("Tracks deltas")
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            if np.isfinite(heat[yi, xi]):
                text_color = "white" if heat[yi, xi] < np.nanmedian(finite) else "black"
                ax1.text(xi, yi, f"{heat[yi, xi]:.1f}", ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)

    rraa = best_per_rraa(df)
    x = np.arange(len(rraa))
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(x, rraa["mean_primary_metric"], width=0.58, color="#2563eb", label="mean trans ratio")
    ax2.errorbar(
        x,
        rraa["mean_primary_metric"],
        yerr=rraa["std_primary_metric"],
        fmt="none",
        ecolor="#1f2937",
        elinewidth=1.3,
        capsize=3,
        label="std",
    )
    ax2.plot(x, rraa["worst_primary_metric"], color="#dc2626", marker="o", lw=2.0, label="worst trans ratio")
    for bar, label in zip(bars, rraa["deltas_tracks"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"T {label}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax2.set_title("RRAA span controls stability")
    ax2.set_xticks(x, rraa["deltas_rraa"], rotation=35, ha="right")
    ax2.set_xlabel("Best protocol per RRAA span")
    ax2.set_ylabel("Translation ratio (lower is better)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    png = out_dir / "stage1_graph_span_search.png"
    pdf = out_dir / "stage1_graph_span_search.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    print(f"saved: {png}")
    print(f"saved: {pdf}")
    print(f"saved: {out_dir / 'stage1_graph_span_unique_protocols.csv'}")


if __name__ == "__main__":
    main()
