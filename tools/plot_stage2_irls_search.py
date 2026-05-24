#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def best_per_protocol(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["irls_iters", "irls_huber_k"]
    idx = df.groupby(cols)["score"].idxmin()
    return df.loc[idx].sort_values("score").reset_index(drop=True)


def best_per_iter(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("irls_iters")["score"].idxmin()
    return df.loc[idx].sort_values("irls_iters").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Stage-2 IRLS search results.")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Stage 2 IRLS search")
    ap.add_argument("--window", type=int, default=4, help="Rolling window for local mean/RMS curves.")
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
            "irls_iters",
            "irls_huber_k",
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
    unique.to_csv(out_dir / "stage2_irls_unique_protocols.csv", index=False)

    best = df.loc[df["score"].idxmin()]
    best_label = f"IRLS {int(best['irls_iters'])}, k={best['irls_huber_k']:g}"

    fig = plt.figure(figsize=(14.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.35])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(df["trial_number"], df["score"], s=34, color="#9aa3ad", alpha=0.75, label="trial")
    ax0.plot(df["trial_number"], df["rolling_mean"], color="#2563eb", lw=1.8, label=f"{window}-trial mean")
    ax0.plot(
        df["trial_number"],
        df["rolling_rms"],
        color="#16a34a",
        lw=1.8,
        linestyle="--",
        label=f"{window}-trial RMS",
    )
    ax0.plot(df["trial_number"], df["best_so_far"], color="#b42318", lw=2.2, label="best so far")
    ax0.scatter([best["trial_number"]], [best["score"]], s=80, color="#b42318", zorder=5)
    ax0.annotate(
        f"best: {best_label}",
        xy=(best["trial_number"], best["score"]),
        xytext=(0.38, 0.35),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"),
        fontsize=9,
    )
    ax0.set_title("Optimization history")
    ax0.set_xlabel("Trial")
    ax0.set_ylabel("Search score (lower is better)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, loc="upper right", fontsize=8)

    iter_order = sorted(df["irls_iters"].dropna().unique())
    k_order = sorted(df["irls_huber_k"].dropna().unique())
    heat = np.full((len(iter_order), len(k_order)), np.nan)
    for _, row in unique.iterrows():
        yi = iter_order.index(row["irls_iters"])
        xi = k_order.index(row["irls_huber_k"])
        heat[yi, xi] = row["score"]

    ax1 = fig.add_subplot(gs[0, 1])
    finite = heat[np.isfinite(heat)]
    im = ax1.imshow(
        heat,
        cmap="viridis_r",
        vmin=float(np.nanmin(finite)),
        vmax=float(np.nanmax(finite)),
        aspect="auto",
    )
    ax1.set_title("Best score by IRLS setting")
    ax1.set_xticks(range(len(k_order)), [f"{k:g}" for k in k_order])
    ax1.set_yticks(range(len(iter_order)), [f"{int(v)}" for v in iter_order])
    ax1.set_xlabel("Huber k")
    ax1.set_ylabel("IRLS iterations")
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            if np.isfinite(heat[yi, xi]):
                text_color = "white" if heat[yi, xi] < np.nanmedian(finite) else "black"
                ax1.text(xi, yi, f"{heat[yi, xi]:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)

    by_iter = best_per_iter(df)
    x = np.arange(len(by_iter))
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(x, by_iter["mean_primary_metric"], width=0.58, color="#2563eb", label="mean trans ratio")
    ax2.errorbar(
        x,
        by_iter["mean_primary_metric"],
        yerr=by_iter["std_primary_metric"],
        fmt="none",
        ecolor="#1f2937",
        elinewidth=1.3,
        capsize=3,
        label="std",
    )
    ax2.plot(x, by_iter["worst_primary_metric"], color="#dc2626", marker="o", lw=2.0, label="worst trans ratio")
    for bar, k in zip(bars, by_iter["irls_huber_k"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"k={k:g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_title("IRLS stabilizes translation")
    ax2.set_xticks(x, [f"{int(v)} iter" for v in by_iter["irls_iters"]])
    ax2.set_xlabel("Best protocol per iteration count")
    ax2.set_ylabel("Translation ratio (lower is better)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    png = out_dir / "stage2_irls_search.png"
    pdf = out_dir / "stage2_irls_search.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    print(f"saved: {png}")
    print(f"saved: {pdf}")
    print(f"saved: {out_dir / 'stage2_irls_unique_protocols.csv'}")


if __name__ == "__main__":
    main()
