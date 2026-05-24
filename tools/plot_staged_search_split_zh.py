#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RRAA_ORDER = ["1,2,3", "1,2,3,5", "1,2,3,5,8", "1,2,3,5,8,13"]
TRACKS_ORDER = ["1", "1,2,3", "1,2,3,5"]


def setup_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def to_num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")


def best_per_protocol_stage1(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["deltas_tracks", "deltas_rraa"])["score"].idxmin()
    return df.loc[idx].sort_values("score").reset_index(drop=True)


def best_per_rraa(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("deltas_rraa")["score"].idxmin()
    out = df.loc[idx].copy()
    out["deltas_rraa"] = pd.Categorical(out["deltas_rraa"], RRAA_ORDER, ordered=True)
    return out.sort_values("deltas_rraa").reset_index(drop=True)


def plot_stage1(trials_csv: Path, out_dir: Path, window: int) -> None:
    df = pd.read_csv(trials_csv)
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
    unique = best_per_protocol_stage1(df)
    best = df.loc[df["score"].idxmin()]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(df["trial_number"], df["score"], s=34, color="#9aa3ad", alpha=0.75, label="单次试验")
    ax.plot(df["trial_number"], df["rolling_mean"], color="#2563eb", lw=1.9, label=f"{window}次滚动均值")
    ax.plot(
        df["trial_number"],
        df["rolling_rms"],
        color="#16a34a",
        lw=1.9,
        linestyle="--",
        label=f"{window}次滚动均方根",
    )
    ax.plot(df["trial_number"], df["best_so_far"], color="#b42318", lw=2.3, label="当前最优")
    ax.scatter([best["trial_number"]], [best["score"]], s=82, color="#b42318", zorder=5)
    ax.annotate(
        f"最优: 轨迹构建跨度 {best['deltas_tracks']}\n旋转平均图跨度 {best['deltas_rraa']}",
        xy=(best["trial_number"], best["score"]),
        xytext=(0.6, -0.2),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.92),
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlabel("试验编号")
    ax.set_ylabel("Score（log，越低越好）")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, out_dir, "stage1_optimization_history_zh")

    heat = np.full((len(TRACKS_ORDER), len(RRAA_ORDER)), np.nan)
    for _, row in unique.iterrows():
        if row["deltas_tracks"] in TRACKS_ORDER and row["deltas_rraa"] in RRAA_ORDER:
            yi = TRACKS_ORDER.index(row["deltas_tracks"])
            xi = RRAA_ORDER.index(row["deltas_rraa"])
            heat[yi, xi] = row["score"]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    finite = heat[np.isfinite(heat)]
    im = ax.imshow(
        heat,
        cmap="viridis_r",
        vmin=float(np.nanmin(finite)),
        vmax=float(np.nanpercentile(finite, 85)),
        aspect="auto",
    )
    ax.set_xticks(range(len(RRAA_ORDER)), RRAA_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(TRACKS_ORDER)), TRACKS_ORDER)
    ax.set_xlabel("旋转平均图跨度")
    ax.set_ylabel("轨迹构建跨度")
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            if np.isfinite(heat[yi, xi]):
                text_color = "white" if heat[yi, xi] < np.nanmedian(finite) else "black"
                ax.text(xi, yi, f"{heat[yi, xi]:.1f}", ha="center", va="center", fontsize=10, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Score（越低越好）")
    save_fig(fig, out_dir, "stage1_graph_span_heatmap_zh")

    rraa = best_per_rraa(df)
    x = np.arange(len(rraa))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(x, rraa["mean_primary_metric"], width=0.58, color="#2563eb", label="平均平移比值")
    ax.errorbar(
        x,
        rraa["mean_primary_metric"],
        yerr=rraa["std_primary_metric"],
        fmt="none",
        ecolor="#1f2937",
        elinewidth=1.3,
        capsize=3,
        label="标准差",
    )
    ax.plot(x, rraa["worst_primary_metric"], color="#dc2626", marker="o", lw=2.0, label="最差平移比值")
    for i, (bar, label) in enumerate(zip(bars, rraa["deltas_tracks"])):
        x_offsets = [0.40, 0.52, 0.48, 0.56]
        y_offsets = [-26.0, -56.0, -24.0, -26.0]

        ax.text(
            bar.get_x() + bar.get_width() * x_offsets[i],
            bar.get_height() + y_offsets[i],
            f"轨迹构建跨度 {label}",
            ha="left",
            va="bottom",
            fontsize=8,
            rotation=0,
            color="black",
        )
    ax.set_xticks(x, rraa["deltas_rraa"], rotation=30, ha="right")
    ax.set_xlim(-0.55, len(rraa) + 0.65)
    ax.set_xlabel("每个旋转平均图跨度下的最优配置")
    ax.set_ylabel("平移归一化比值（越低越好）")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, out_dir, "stage1_rraa_span_stability_zh")


def best_per_protocol_stage2(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["irls_iters", "irls_huber_k"])["score"].idxmin()
    return df.loc[idx].sort_values("score").reset_index(drop=True)


def best_per_iter(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("irls_iters")["score"].idxmin()
    return df.loc[idx].sort_values("irls_iters").reset_index(drop=True)


def plot_stage2(trials_csv: Path, out_dir: Path, window: int) -> None:
    df = pd.read_csv(trials_csv)
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
    unique = best_per_protocol_stage2(df)
    best = df.loc[df["score"].idxmin()]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(df["trial_number"], df["score"], s=34, color="#9aa3ad", alpha=0.75, label="单次试验")
    ax.plot(df["trial_number"], df["rolling_mean"], color="#2563eb", lw=1.9, label=f"{window}次滚动均值")
    ax.plot(
        df["trial_number"],
        df["rolling_rms"],
        color="#16a34a",
        lw=1.9,
        linestyle="--",
        label=f"{window}次滚动均方根",
    )
    ax.plot(df["trial_number"], df["best_so_far"], color="#b42318", lw=2.3, label="当前最优")
    ax.scatter([best["trial_number"]], [best["score"]], s=82, color="#b42318", zorder=5)
    ax.annotate(
        f"最优: IRLS {int(best['irls_iters'])}, k={best['irls_huber_k']:g}",
        xy=(best["trial_number"], best["score"]),
        xytext=(0.48, -0.18),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.92),
        fontsize=10,
        annotation_clip=False,
    )
    ax.set_xlabel("试验编号")
    ax.set_ylabel("Score（越低越好）")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, out_dir, "stage2_optimization_history_zh")

    iter_order = sorted(df["irls_iters"].dropna().unique())
    k_order = sorted(df["irls_huber_k"].dropna().unique())
    heat = np.full((len(iter_order), len(k_order)), np.nan)
    for _, row in unique.iterrows():
        yi = iter_order.index(row["irls_iters"])
        xi = k_order.index(row["irls_huber_k"])
        heat[yi, xi] = row["score"]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    finite = heat[np.isfinite(heat)]
    im = ax.imshow(
        heat,
        cmap="viridis_r",
        vmin=float(np.nanmin(finite)),
        vmax=float(np.nanmax(finite)),
        aspect="auto",
    )
    ax.set_xticks(range(len(k_order)), [f"{k:g}" for k in k_order])
    ax.set_yticks(range(len(iter_order)), [f"{int(v)}" for v in iter_order])
    ax.set_xlabel("Huber k")
    ax.set_ylabel("IRLS迭代次数")
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            if np.isfinite(heat[yi, xi]):
                text_color = "white" if heat[yi, xi] < np.nanmedian(finite) else "black"
                ax.text(xi, yi, f"{heat[yi, xi]:.2f}", ha="center", va="center", fontsize=10, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Score（越低越好）")
    save_fig(fig, out_dir, "stage2_irls_heatmap_zh")

    by_iter = best_per_iter(df)
    x = np.arange(len(by_iter))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(x, by_iter["mean_primary_metric"], width=0.58, color="#2563eb", label="平均平移比值")
    ax.errorbar(
        x,
        by_iter["mean_primary_metric"],
        yerr=by_iter["std_primary_metric"],
        fmt="none",
        ecolor="#1f2937",
        elinewidth=1.3,
        capsize=3,
        label="标准差",
    )
    ax.plot(x, by_iter["worst_primary_metric"], color="#dc2626", marker="o", lw=2.0, label="最差平移比值")
    for bar, k in zip(bars, by_iter["irls_huber_k"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.72,
            f"k={k:g}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.set_xticks(x, [f"{int(v)}轮" for v in by_iter["irls_iters"]])
    ax.set_xlabel("每个IRLS迭代次数下的最优配置")
    ax.set_ylabel("平移归一化比值（越低越好）")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, out_dir, "stage2_irls_stability_zh")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export split Chinese staged-search plots.")
    ap.add_argument("--stage1_csv", default="runs/paper_v2/staged_search/stage1_graph_span_dev8_v2/trials_summary.csv")
    ap.add_argument("--stage1_out", default="runs/paper_v2/staged_search/stage1_graph_span_dev8_v2/figures/split_zh")
    ap.add_argument("--stage1_window", type=int, default=6)
    ap.add_argument("--stage2_csv", default="runs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2/trials_summary.csv")
    ap.add_argument("--stage2_out", default="runs/paper_v2/staged_search/stage2_irls_on_A4_uniform_dev8_v2/figures/split_zh")
    ap.add_argument("--stage2_window", type=int, default=4)
    args = ap.parse_args()

    setup_fonts()
    plot_stage1(Path(args.stage1_csv), Path(args.stage1_out), max(1, args.stage1_window))
    plot_stage2(Path(args.stage2_csv), Path(args.stage2_out), max(1, args.stage2_window))


if __name__ == "__main__":
    main()
