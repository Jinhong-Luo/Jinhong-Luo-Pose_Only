#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, ttk


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "paper_v2"
COLMAP_ROOT = REPO_ROOT / "Colmap_runs"
FINAL_REPORT_ROOT = RUNS_ROOT / "final_benchmark_report"

OURS_METHODS = ["gt_ligt", "gt_ligt_pa", "rraa_ligt", "rraa_ligt_pa"]
METHOD_LABELS = {
    "gt_ligt": "GT+LiGT",
    "gt_ligt_pa": "GT+LiGT+PA",
    "rraa_ligt": "RRAA+LiGT",
    "rraa_ligt_pa": "RRAA+LiGT+PA",
    "colmap": "COLMAP",
}
METHOD_COLORS = {
    "gt_ligt": "#2563eb",
    "gt_ligt_pa": "#7c3aed",
    "rraa_ligt": "#f97316",
    "rraa_ligt_pa": "#dc2626",
    "colmap": "#059669",
}

BASE_METRICS = {
    "rotation_median_deg": "Rotation Median (deg)",
    "translation_mm_median": "Translation Median (mm)",
    "translation_mm_p90": "Translation P90 (mm)",
    "runtime_total_sec": "Runtime Total (sec)",
    "peak_memory_mb": "Peak Memory (MB)",
    "runtime_rraa_sec": "RRAA Runtime (sec)",
    "runtime_pose_only_sec": "Pose-Only Runtime (sec)",
    "fair_backend_time_sec": "Backend Core / Mapper Time (sec)",
    "fair_backend_peak_memory_mb": "Backend Core / Mapper Peak Memory (MB)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_dataset_name(value: str) -> str:
    return "strecha" if value.lower() == "strecha" else value


def normalize_colmap_scene(dataset: str, scene: str) -> str:
    if dataset == "strecha":
        mapping = {
            "castle-p19": "Castle-P19",
            "castle-p30": "Castle-P30",
            "entry-p10": "entry-P10",
            "fountain-p11": "fountain-P11",
            "herz-jesus-p25": "Herz-Jesus-P25",
            "herz-jesus-p8": "Herz-Jesus-P8",
        }
        return mapping.get(scene.lower(), scene)
    return scene


@dataclass
class BenchmarkRow:
    source_type: str
    source_name: str
    dataset: str
    scene: str
    method: str
    label: str
    metrics: dict[str, float] = field(default_factory=dict)
    step_time: dict[str, float] = field(default_factory=dict)
    step_memory: dict[str, float] = field(default_factory=dict)
    run_dir: str = ""


class BenchmarkRepository:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.rows: list[BenchmarkRow] = []
        self.datasets: list[str] = []
        self.methods: list[str] = []
        self.scenes_by_dataset: dict[str, list[str]] = {}
        self.ours_step_names: list[str] = []
        self.colmap_step_names: list[str] = []

    def load(self) -> None:
        self.rows = []
        self._load_ours_rows()
        self._load_colmap_rows()
        self.datasets = sorted({r.dataset for r in self.rows})
        self.methods = [m for m in OURS_METHODS + ["colmap"] if any(r.method == m for r in self.rows)]
        self.scenes_by_dataset = {
            dataset: sorted({r.scene for r in self.rows if r.dataset == dataset})
            for dataset in self.datasets
        }
        self.ours_step_names = sorted(
            {
                step
                for row in self.rows
                if row.source_type == "ours"
                for step in row.step_time.keys() | row.step_memory.keys()
            }
        )
        self.colmap_step_names = sorted(
            {
                step
                for row in self.rows
                if row.source_type == "colmap"
                for step in row.step_time.keys() | row.step_memory.keys()
            }
        )

    def _load_ours_rows(self) -> None:
        latest: dict[tuple[str, str, str], tuple[float, BenchmarkRow]] = {}
        for eval_dir in sorted(RUNS_ROOT.glob("eval_*")):
            csv_path = eval_dir / "paper_summary_main.csv"
            if not csv_path.exists():
                continue
            try:
                rows = read_csv(csv_path)
            except Exception:
                continue
            mtime = csv_path.stat().st_mtime
            for row in rows:
                if row.get("status") != "ok":
                    continue
                method = row.get("experiment_group", "")
                if method not in OURS_METHODS:
                    continue
                dataset = normalize_dataset_name(row.get("dataset", ""))
                scene = row.get("scene", "")
                metrics = {
                    key: to_float(row.get(key))
                    for key in BASE_METRICS
                }
                summary_path = Path(row.get("run_dir", "")) / "experiment_summary.json" if row.get("run_dir") else None
                step_time = {}
                step_memory = {}
                if summary_path and summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                        step_time = {
                            k: to_float(v)
                            for k, v in (summary.get("runtime_by_step_sec") or {}).items()
                        }
                        step_memory = {
                            k: to_float(v)
                            for k, v in (summary.get("peak_memory_by_step_mb") or {}).items()
                        }
                    except Exception:
                        pass
                bench_row = BenchmarkRow(
                    source_type="ours",
                    source_name=eval_dir.name,
                    dataset=dataset,
                    scene=scene,
                    method=method,
                    label=METHOD_LABELS.get(method, method),
                    metrics=metrics,
                    step_time=step_time,
                    step_memory=step_memory,
                    run_dir=row.get("run_dir", ""),
                )
                key = (dataset, scene, method)
                prev = latest.get(key)
                if prev is None or mtime >= prev[0]:
                    latest[key] = (mtime, bench_row)
        self.rows.extend(item[1] for item in latest.values())

    def _load_colmap_rows(self) -> None:
        precision_by_key: dict[tuple[str, str], dict[str, float]] = {}
        cache_root = FINAL_REPORT_ROOT / "colmap_eval_cache"
        if cache_root.exists():
            for summary_path in cache_root.glob("*/*/summary.json"):
                try:
                    obj = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                dataset = normalize_dataset_name(obj.get("dataset", ""))
                scene = obj.get("scene", "")
                precision_by_key[(dataset, scene)] = {
                    "rotation_median_deg": to_float(obj.get("rotation_median_deg")),
                    "translation_mm_median": to_float(obj.get("translation_mm_median")),
                    "translation_mm_p90": to_float(obj.get("translation_mm_p90")),
                }

        runtime_csv = COLMAP_ROOT / "colmap_runtime_memory_summary.csv"
        if not runtime_csv.exists():
            return

        stage_rows: dict[tuple[str, str], dict[str, float]] = {}
        stage_mem_rows: dict[tuple[str, str], dict[str, float]] = {}
        total_metrics: dict[tuple[str, str], dict[str, float]] = {}
        for row in read_csv(runtime_csv):
            dataset = normalize_dataset_name(row.get("Dataset", ""))
            scene = normalize_colmap_scene(dataset, row.get("Scene", ""))
            stage = row.get("Stage", "")
            key = (dataset, scene)
            if stage == "TOTAL":
                total_metrics[key] = {
                    "runtime_total_sec": to_float(row.get("TimeSeconds")),
                    "peak_memory_mb": to_float(row.get("PeakObservedWorkingSetGB")) * 1024.0,
                }
            else:
                stage_rows.setdefault(key, {})[stage] = to_float(row.get("TimeSeconds"))
                stage_mem_rows.setdefault(key, {})[stage] = to_float(row.get("PeakObservedWorkingSetGB")) * 1024.0

        all_keys = sorted(set(total_metrics) | set(precision_by_key))
        for key in all_keys:
            dataset, scene = key
            metrics = {}
            metrics.update({k: float("nan") for k in BASE_METRICS})
            metrics.update(total_metrics.get(key, {}))
            metrics.update(precision_by_key.get(key, {}))
            self.rows.append(
                BenchmarkRow(
                    source_type="colmap",
                    source_name="Colmap_runs",
                    dataset=dataset,
                    scene=scene,
                    method="colmap",
                    label=METHOD_LABELS["colmap"],
                    metrics=metrics,
                    step_time=stage_rows.get(key, {}),
                    step_memory=stage_mem_rows.get(key, {}),
                    run_dir="",
                )
            )

    def metric_choices(self) -> dict[str, str]:
        choices = dict(BASE_METRICS)
        for step in self.ours_step_names:
            choices[f"ours_time:{step}"] = f"Ours Time / {step}"
            choices[f"ours_memory:{step}"] = f"Ours Memory / {step}"
        for step in self.colmap_step_names:
            choices[f"colmap_time:{step}"] = f"COLMAP Time / {step}"
            choices[f"colmap_memory:{step}"] = f"COLMAP Memory / {step}"
        return choices

    def value_for(self, row: BenchmarkRow, metric_key: str) -> float:
        if metric_key == "fair_backend_time_sec":
            if row.source_type == "colmap":
                return to_float(row.step_time.get("mapper"))
            pose_only = to_float(row.step_time.get("pose_only"))
            rraa = to_float(row.step_time.get("rraa"))
            total = 0.0
            has_any = False
            if not math.isnan(pose_only):
                total += pose_only
                has_any = True
            if row.method.startswith("rraa_") and not math.isnan(rraa):
                total += rraa
                has_any = True
            return total if has_any else float("nan")
        if metric_key == "fair_backend_peak_memory_mb":
            if row.source_type == "colmap":
                return to_float(row.step_memory.get("mapper"))
            vals = []
            pose_only = to_float(row.step_memory.get("pose_only"))
            if not math.isnan(pose_only):
                vals.append(pose_only)
            if row.method.startswith("rraa_"):
                rraa = to_float(row.step_memory.get("rraa"))
                if not math.isnan(rraa):
                    vals.append(rraa)
            return max(vals) if vals else float("nan")
        if metric_key in BASE_METRICS:
            return to_float(row.metrics.get(metric_key))
        if metric_key.startswith("ours_time:"):
            return to_float(row.step_time.get(metric_key.split(":", 1)[1])) if row.source_type == "ours" else float("nan")
        if metric_key.startswith("ours_memory:"):
            return to_float(row.step_memory.get(metric_key.split(":", 1)[1])) if row.source_type == "ours" else float("nan")
        if metric_key.startswith("colmap_time:"):
            return to_float(row.step_time.get(metric_key.split(":", 1)[1])) if row.source_type == "colmap" else float("nan")
        if metric_key.startswith("colmap_memory:"):
            return to_float(row.step_memory.get(metric_key.split(":", 1)[1])) if row.source_type == "colmap" else float("nan")
        return float("nan")

    def filter_rows(self, datasets: list[str], scenes: list[str], methods: list[str]) -> list[BenchmarkRow]:
        selected = []
        for row in self.rows:
            if datasets and row.dataset not in datasets:
                continue
            if scenes and row.scene not in scenes:
                continue
            if methods and row.method not in methods:
                continue
            selected.append(row)
        return selected


class BenchmarkGUI(tk.Tk):
    def __init__(self, repo: BenchmarkRepository):
        super().__init__()
        self.repo = repo
        self.title("Benchmark Result Visualizer")
        self.geometry("1580x980")
        self.minsize(1320, 860)

        self.metric_map = self.repo.metric_choices()
        self.dataset_listbox: tk.Listbox
        self.scene_listbox: tk.Listbox
        self.method_listbox: tk.Listbox
        self.metric_combo: ttk.Combobox
        self.plot_mode_combo: ttk.Combobox
        self.scale_combo: ttk.Combobox
        self.compare_mode_combo: ttk.Combobox
        self.breakdown_metric_combo: ttk.Combobox
        self.breakdown_dataset_combo: ttk.Combobox
        self.breakdown_scene_combo: ttk.Combobox
        self.breakdown_method_combo: ttk.Combobox
        self.status_var = tk.StringVar(value="Ready")

        self.figure = Figure(figsize=(8.5, 5.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.breakdown_figure = Figure(figsize=(8.5, 5.4), dpi=100)
        self.breakdown_ax = self.breakdown_figure.add_subplot(111)

        self._build_ui()
        self._populate_controls()
        self.refresh_views()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Datasets").grid(row=0, column=0, sticky="w")
        self.dataset_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, exportselection=False, height=8)
        self.dataset_listbox.grid(row=1, column=0, sticky="ew")
        self.dataset_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_dataset_change())

        ttk.Label(left, text="Scenes").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.scene_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, exportselection=False, height=12)
        self.scene_listbox.grid(row=3, column=0, sticky="ew")

        ttk.Label(left, text="Methods").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.method_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, exportselection=False, height=7)
        self.method_listbox.grid(row=5, column=0, sticky="ew")

        ttk.Label(left, text="Metric").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.metric_combo = ttk.Combobox(left, state="readonly")
        self.metric_combo.grid(row=7, column=0, sticky="ew")
        self.metric_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_views())

        ttk.Label(left, text="Plot Mode").grid(row=8, column=0, sticky="w", pady=(10, 0))
        self.plot_mode_combo = ttk.Combobox(left, state="readonly", values=["scene_grouped", "method_average"])
        self.plot_mode_combo.grid(row=9, column=0, sticky="ew")
        self.plot_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_views())

        ttk.Label(left, text="Y Scale").grid(row=10, column=0, sticky="w", pady=(10, 0))
        self.scale_combo = ttk.Combobox(left, state="readonly", values=["linear", "log"])
        self.scale_combo.grid(row=11, column=0, sticky="ew")
        self.scale_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_views())

        ttk.Label(left, text="Compare Scope").grid(row=12, column=0, sticky="w", pady=(10, 0))
        self.compare_mode_combo = ttk.Combobox(left, state="readonly", values=["Full pipeline", "Backend-only fair compare"])
        self.compare_mode_combo.grid(row=13, column=0, sticky="ew")
        self.compare_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_compare_mode_change())

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=14, column=0, sticky="ew", pady=(14, 0))
        btn_frame.columnconfigure((0, 1), weight=1)
        ttk.Button(btn_frame, text="Refresh Data", command=self.reload_data).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btn_frame, text="Export Chart", command=self.export_current_chart).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Button(left, text="Refresh View", command=self.refresh_views).grid(row=15, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Notebook(self)
        right.grid(row=0, column=1, sticky="nsew")

        compare_tab = ttk.Frame(right, padding=8)
        compare_tab.columnconfigure(0, weight=1)
        compare_tab.rowconfigure(0, weight=1)
        compare_tab.rowconfigure(1, weight=0)
        compare_tab.rowconfigure(2, weight=1)
        right.add(compare_tab, text="Compare")

        canvas_frame = ttk.Frame(compare_tab)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        self.table = ttk.Treeview(compare_tab, show="headings", height=16)
        self.table.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        table_scroll = ttk.Scrollbar(compare_tab, orient="vertical", command=self.table.yview)
        table_scroll.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        self.table.configure(yscrollcommand=table_scroll.set)

        breakdown_tab = ttk.Frame(right, padding=8)
        breakdown_tab.columnconfigure(0, weight=0)
        breakdown_tab.columnconfigure(1, weight=1)
        breakdown_tab.rowconfigure(0, weight=1)
        right.add(breakdown_tab, text="Step Breakdown")

        ctl = ttk.Frame(breakdown_tab)
        ctl.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        ctl.columnconfigure(0, weight=1)

        ttk.Label(ctl, text="Dataset").grid(row=0, column=0, sticky="w")
        self.breakdown_dataset_combo = ttk.Combobox(ctl, state="readonly")
        self.breakdown_dataset_combo.grid(row=1, column=0, sticky="ew")
        self.breakdown_dataset_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_breakdown_dataset_change())

        ttk.Label(ctl, text="Scene").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.breakdown_scene_combo = ttk.Combobox(ctl, state="readonly")
        self.breakdown_scene_combo.grid(row=3, column=0, sticky="ew")
        self.breakdown_scene_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_breakdown())

        ttk.Label(ctl, text="Method").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.breakdown_method_combo = ttk.Combobox(ctl, state="readonly")
        self.breakdown_method_combo.grid(row=5, column=0, sticky="ew")
        self.breakdown_method_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_breakdown())

        ttk.Label(ctl, text="Breakdown Type").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.breakdown_metric_combo = ttk.Combobox(ctl, state="readonly", values=["time", "memory"])
        self.breakdown_metric_combo.grid(row=7, column=0, sticky="ew")
        self.breakdown_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_breakdown())

        ttk.Button(ctl, text="Export Breakdown", command=self.export_breakdown_chart).grid(row=8, column=0, sticky="ew", pady=(12, 0))

        chart_frame = ttk.Frame(breakdown_tab)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.breakdown_canvas = FigureCanvasTkAgg(self.breakdown_figure, master=chart_frame)
        self.breakdown_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        breakdown_toolbar = NavigationToolbar2Tk(self.breakdown_canvas, chart_frame, pack_toolbar=False)
        breakdown_toolbar.update()
        breakdown_toolbar.grid(row=1, column=0, sticky="ew")

        bottom = ttk.Frame(self, padding=(10, 4))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    def _populate_controls(self) -> None:
        self.dataset_listbox.delete(0, tk.END)
        for dataset in self.repo.datasets:
            self.dataset_listbox.insert(tk.END, dataset)
        for idx in range(len(self.repo.datasets)):
            self.dataset_listbox.selection_set(idx)

        self.method_listbox.delete(0, tk.END)
        for method in self.repo.methods:
            self.method_listbox.insert(tk.END, METHOD_LABELS.get(method, method))
        for idx in range(len(self.repo.methods)):
            self.method_listbox.selection_set(idx)

        self.metric_map = self.repo.metric_choices()
        metric_labels = list(self.metric_map.values())
        self.metric_keys = list(self.metric_map.keys())
        self.metric_combo["values"] = metric_labels
        if metric_labels:
            self.metric_combo.current(1 if len(metric_labels) > 1 else 0)

        self.plot_mode_combo.current(0)
        self.scale_combo.current(1)
        self.compare_mode_combo.current(0)
        self.breakdown_metric_combo.current(0)

        self.breakdown_dataset_combo["values"] = self.repo.datasets
        if self.repo.datasets:
            self.breakdown_dataset_combo.current(0)
        self._on_dataset_change()
        self._on_breakdown_dataset_change()
        self._on_compare_mode_change()

    def selected_datasets(self) -> list[str]:
        return [self.dataset_listbox.get(i) for i in self.dataset_listbox.curselection()]

    def selected_scenes(self) -> list[str]:
        return [self.scene_listbox.get(i) for i in self.scene_listbox.curselection()]

    def selected_methods(self) -> list[str]:
        idxs = self.method_listbox.curselection()
        return [self.repo.methods[i] for i in idxs]

    def selected_metric_key(self) -> str:
        idx = self.metric_combo.current()
        if idx < 0:
            return "translation_mm_median"
        return self.metric_keys[idx]

    def _metric_keys_for_compare_mode(self, compare_mode: str) -> list[str]:
        base = [
            "rotation_median_deg",
            "translation_mm_median",
            "translation_mm_p90",
            "runtime_total_sec",
            "peak_memory_mb",
            "runtime_rraa_sec",
            "runtime_pose_only_sec",
            "fair_backend_time_sec",
            "fair_backend_peak_memory_mb",
        ]
        if compare_mode == "Backend-only fair compare":
            return [
                "rotation_median_deg",
                "translation_mm_median",
                "translation_mm_p90",
                "fair_backend_time_sec",
                "fair_backend_peak_memory_mb",
            ]
        return base

    def _on_compare_mode_change(self) -> None:
        compare_mode = self.compare_mode_combo.get() or "Full pipeline"
        allowed_keys = self._metric_keys_for_compare_mode(compare_mode)
        current_key = self.selected_metric_key()
        self.metric_keys = allowed_keys
        self.metric_combo["values"] = [self.metric_map[k] for k in allowed_keys]
        if current_key in allowed_keys:
            self.metric_combo.current(allowed_keys.index(current_key))
        else:
            default_key = "fair_backend_time_sec" if compare_mode == "Backend-only fair compare" else "translation_mm_median"
            self.metric_combo.current(allowed_keys.index(default_key))
        self.refresh_views()

    def _on_dataset_change(self) -> None:
        current_scenes = set(self.selected_scenes())
        self.scene_listbox.delete(0, tk.END)
        scenes = []
        for dataset in self.selected_datasets():
            scenes.extend(self.repo.scenes_by_dataset.get(dataset, []))
        for scene in sorted(dict.fromkeys(scenes)):
            self.scene_listbox.insert(tk.END, scene)
        for idx in range(self.scene_listbox.size()):
            if self.scene_listbox.get(idx) in current_scenes or not current_scenes:
                self.scene_listbox.selection_set(idx)
        self.refresh_views()

    def _on_breakdown_dataset_change(self) -> None:
        dataset = self.breakdown_dataset_combo.get()
        scenes = self.repo.scenes_by_dataset.get(dataset, [])
        self.breakdown_scene_combo["values"] = scenes
        if scenes:
            self.breakdown_scene_combo.current(0)
        methods = sorted({row.method for row in self.repo.rows if row.dataset == dataset})
        method_labels = [METHOD_LABELS.get(m, m) for m in methods]
        self.breakdown_method_values = methods
        self.breakdown_method_combo["values"] = method_labels
        if method_labels:
            self.breakdown_method_combo.current(0)
        self.refresh_breakdown()

    def reload_data(self) -> None:
        self.repo.load()
        self._populate_controls()
        self.status_var.set(f"Reloaded {len(self.repo.rows)} rows.")

    def refresh_views(self) -> None:
        datasets = self.selected_datasets()
        scenes = self.selected_scenes()
        methods = self.selected_methods()
        metric_key = self.selected_metric_key()
        rows = self.repo.filter_rows(datasets, scenes, methods)
        self._draw_compare(rows, metric_key, self.plot_mode_combo.get(), self.scale_combo.get())
        self._fill_table(rows, metric_key)
        self.status_var.set(
            f"{len(rows)} rows | mode={self.compare_mode_combo.get()} | datasets={','.join(datasets) if datasets else 'all'} | metric={self.metric_map.get(metric_key, metric_key)}"
        )
        self.refresh_breakdown()

    def _draw_compare(self, rows: list[BenchmarkRow], metric_key: str, plot_mode: str, scale: str) -> None:
        self.ax.clear()
        if not rows:
            self.ax.text(0.5, 0.5, "No rows selected.", ha="center", va="center")
            self.canvas.draw_idle()
            return

        if plot_mode == "method_average":
            methods = [m for m in self.selected_methods() if any(r.method == m for r in rows)]
            values = []
            for method in methods:
                vals = [self.repo.value_for(r, metric_key) for r in rows if r.method == method]
                vals = [v for v in vals if not math.isnan(v)]
                values.append(sum(vals) / len(vals) if vals else float("nan"))
            colors = [METHOD_COLORS.get(m, "#6b7280") for m in methods]
            self.ax.bar(
                [METHOD_LABELS.get(m, m) for m in methods],
                values,
                color=colors,
                edgecolor="black",
                linewidth=0.8,
            )
            self.ax.set_title("Method Average")
            self.ax.set_xlabel("Method")
        else:
            scene_keys = sorted({(r.dataset, r.scene) for r in rows})
            methods = [m for m in self.selected_methods() if any(r.method == m for r in rows)]
            width = 0.8 / max(len(methods), 1)
            x = list(range(len(scene_keys)))
            for idx, method in enumerate(methods):
                vals = []
                for dataset, scene in scene_keys:
                    match = next((r for r in rows if r.dataset == dataset and r.scene == scene and r.method == method), None)
                    vals.append(self.repo.value_for(match, metric_key) if match else float("nan"))
                xpos = [v - 0.4 + width / 2 + idx * width for v in x]
                self.ax.bar(
                    xpos,
                    vals,
                    width=width,
                    color=METHOD_COLORS.get(method, "#6b7280"),
                    label=METHOD_LABELS.get(method, method),
                    edgecolor="black",
                    linewidth=0.8,
                )
            self.ax.set_xticks(x)
            self.ax.set_xticklabels([f"{d}/{s}" for d, s in scene_keys], rotation=25, ha="right")
            self.ax.set_title("Per-Scene Comparison")
            self.ax.legend(ncol=2)
            self.ax.set_xlabel("Scene")

        self.ax.set_ylabel(self.metric_map.get(metric_key, metric_key))
        if scale == "log":
            positives = []
            for row in rows:
                v = self.repo.value_for(row, metric_key)
                if not math.isnan(v) and v > 0:
                    positives.append(v)
            if positives:
                self.ax.set_yscale("log")
        self.ax.grid(axis="y", alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _fill_table(self, rows: list[BenchmarkRow], metric_key: str) -> None:
        columns = ("dataset", "scene", "method", "value", "run_dir")
        self.table.delete(*self.table.get_children())
        self.table["columns"] = columns
        for col, width in [("dataset", 90), ("scene", 140), ("method", 120), ("value", 110), ("run_dir", 620)]:
            self.table.heading(col, text=col)
            self.table.column(col, width=width, anchor="w")
        rows_sorted = sorted(rows, key=lambda r: (r.dataset, r.scene, r.method))
        for row in rows_sorted:
            value = self.repo.value_for(row, metric_key)
            self.table.insert(
                "",
                tk.END,
                values=(
                    row.dataset,
                    row.scene,
                    METHOD_LABELS.get(row.method, row.method),
                    "" if math.isnan(value) else f"{value:.6g}",
                    row.run_dir or row.source_name,
                ),
            )

    def refresh_breakdown(self) -> None:
        self.breakdown_ax.clear()
        dataset = self.breakdown_dataset_combo.get()
        scene = self.breakdown_scene_combo.get()
        method_idx = self.breakdown_method_combo.current()
        if not dataset or not scene or method_idx < 0:
            self.breakdown_ax.text(0.5, 0.5, "Select dataset / scene / method.", ha="center", va="center")
            self.breakdown_canvas.draw_idle()
            return
        method = self.breakdown_method_values[method_idx]
        row = next((r for r in self.repo.rows if r.dataset == dataset and r.scene == scene and r.method == method), None)
        if row is None:
            self.breakdown_ax.text(0.5, 0.5, "No row found.", ha="center", va="center")
            self.breakdown_canvas.draw_idle()
            return

        if self.breakdown_metric_combo.get() == "memory":
            items = row.step_memory
            ylabel = "Peak Memory (MB)"
        else:
            items = row.step_time
            ylabel = "Runtime (sec)"

        if not items:
            self.breakdown_ax.text(0.5, 0.5, "No per-step data for this method.", ha="center", va="center")
            self.breakdown_canvas.draw_idle()
            return

        keys = list(items.keys())
        vals = [to_float(items[k]) for k in keys]
        x = list(range(len(keys)))
        self.breakdown_ax.bar(x, vals, color=METHOD_COLORS.get(method, "#6b7280"), edgecolor="black", linewidth=0.8)
        self.breakdown_ax.set_title(f"{dataset} / {scene} / {METHOD_LABELS.get(method, method)}")
        self.breakdown_ax.set_ylabel(ylabel)
        self.breakdown_ax.set_xticks(x)
        self.breakdown_ax.set_xticklabels(keys, rotation=25, ha="right")
        self.breakdown_ax.grid(axis="y", alpha=0.25)
        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw_idle()

    def export_current_chart(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Chart",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.figure.savefig(path, dpi=180, bbox_inches="tight")
        self.status_var.set(f"Saved chart to {path}")

    def export_breakdown_chart(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Breakdown Chart",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.breakdown_figure.savefig(path, dpi=180, bbox_inches="tight")
        self.status_var.set(f"Saved breakdown chart to {path}")


def dump_summary(repo: BenchmarkRepository) -> None:
    print(
        json.dumps(
            {
                "datasets": repo.datasets,
                "methods": repo.methods,
                "scenes_by_dataset": repo.scenes_by_dataset,
                "ours_step_names": repo.ours_step_names,
                "colmap_step_names": repo.colmap_step_names,
                "row_count": len(repo.rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive GUI for benchmark result comparison.")
    ap.add_argument("--dump-summary", action="store_true", help="Load data and print available datasets / methods / metrics.")
    args = ap.parse_args()

    repo = BenchmarkRepository(REPO_ROOT)
    repo.load()

    if args.dump_summary:
        dump_summary(repo)
        return

    try:
        app = BenchmarkGUI(repo)
    except tk.TclError as exc:
        raise SystemExit(f"Failed to start Tk GUI: {exc}")
    app.mainloop()


if __name__ == "__main__":
    main()
