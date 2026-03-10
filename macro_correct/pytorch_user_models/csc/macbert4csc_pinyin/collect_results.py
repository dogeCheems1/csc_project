#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect metrics.json files and generate summary csv/tables/plots."""

import csv
import json
import os
from collections import defaultdict
from statistics import mean


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
OUT_ROOT = os.path.join(ROOT, "macro_correct", "output", "text_correction")


def find_metrics_files():
    files = []
    for cur_root, _, names in os.walk(OUT_ROOT):
        for name in names:
            if name == "metrics.json":
                files.append(os.path.join(cur_root, name))
    return files


def load_rows():
    rows = []
    for path in find_metrics_files():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "task_name": data.get("task_name"),
                "run_name": data.get("run_name"),
                "dataset_profile": data.get("dataset_profile"),
                "seed": data.get("seed"),
                "use_pinyin": data.get("use_pinyin"),
                "best_cor_f1": data.get("best_cor_f1"),
                "best_step": data.get("best_step"),
                "max_train_steps": data.get("max_train_steps"),
                "run_dir": data.get("run_dir"),
                "history": data.get("history", []),
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_summary(rows):
    return sorted(rows, key=lambda x: (x["task_name"] or "", x["dataset_profile"] or "", str(x["seed"]), str(x["use_pinyin"])))


def filter_rows_by_task(rows, task_name):
    return [r for r in rows if r.get("task_name") == task_name]


def build_ablation(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset_profile"], r["use_pinyin"])].append(r["best_cor_f1"])
    out = []
    for (dataset_profile, use_pinyin), vals in sorted(grouped.items()):
        out.append(
            {
                "dataset_profile": dataset_profile,
                "use_pinyin": use_pinyin,
                "count": len(vals),
                "mean_best_cor_f1": f"{mean(vals):.4f}",
            }
        )
    return out


def build_stability(rows):
    by_dataset_seed = defaultdict(dict)
    for r in rows:
        key = (r["dataset_profile"], r["seed"])
        by_dataset_seed[key][bool(r["use_pinyin"])] = r["best_cor_f1"]
    out = []
    for (dataset_profile, seed), pair in sorted(by_dataset_seed.items()):
        if True in pair and False in pair:
            delta = pair[True] - pair[False]
            out.append(
                {
                    "dataset_profile": dataset_profile,
                    "seed": seed,
                    "baseline_f1": f"{pair[False]:.4f}",
                    "pinyin_f1": f"{pair[True]:.4f}",
                    "delta": f"{delta:+.4f}",
                    "improved": int(delta > 0),
                }
            )
    return out


def plot_curves(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    curves_dir = os.path.join(OUT_ROOT, "plots")
    os.makedirs(curves_dir, exist_ok=True)

    # F1-step curves (one line per run)
    plt.figure(figsize=(10, 6))
    for r in rows:
        hist = r.get("history", [])
        if not hist:
            continue
        x = [h.get("global_step") for h in hist]
        y = [h.get("cor_f1") for h in hist]
        label = f"{r['task_name']}|{r['dataset_profile']}|seed{r['seed']}|pinyin{int(bool(r['use_pinyin']))}"
        plt.plot(x, y, label=label, alpha=0.7)
    plt.xlabel("global_step")
    plt.ylabel("cor_f1")
    plt.title("F1-step Curves")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "f1_step_curves.png"), dpi=150)
    plt.close()

    # Dataset bar chart (mean baseline vs pinyin)
    grouped = defaultdict(lambda: {False: [], True: []})
    for r in rows:
        grouped[r["dataset_profile"]][bool(r["use_pinyin"])].append(r["best_cor_f1"])

    datasets = sorted(grouped.keys())
    base_vals = [mean(grouped[d][False]) if grouped[d][False] else 0.0 for d in datasets]
    py_vals = [mean(grouped[d][True]) if grouped[d][True] else 0.0 for d in datasets]

    x = range(len(datasets))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], base_vals, width, label="baseline")
    plt.bar([i + width / 2 for i in x], py_vals, width, label="pinyin")
    plt.xticks(list(x), datasets, rotation=20)
    plt.ylabel("mean best_cor_f1")
    plt.title("Dataset Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, "dataset_bar.png"), dpi=150)
    plt.close()


def main():
    rows = load_rows()
    if not rows:
        print("No metrics.json found.")
        return

    summary = build_summary(rows)
    ablation = build_ablation(rows)
    stability = build_stability(rows)
    confirm_rows = filter_rows_by_task(rows, "stage1_confirm")
    confirm_summary = build_summary(confirm_rows)
    confirm_ablation = build_ablation(confirm_rows)
    confirm_stability = build_stability(confirm_rows)

    write_csv(
        os.path.join(OUT_ROOT, "results_summary.csv"),
        summary,
        ["task_name", "run_name", "dataset_profile", "seed", "use_pinyin", "best_cor_f1", "best_step", "max_train_steps", "run_dir"],
    )
    write_csv(
        os.path.join(OUT_ROOT, "ablation_table.csv"),
        ablation,
        ["dataset_profile", "use_pinyin", "count", "mean_best_cor_f1"],
    )
    write_csv(
        os.path.join(OUT_ROOT, "stability_table.csv"),
        stability,
        ["dataset_profile", "seed", "baseline_f1", "pinyin_f1", "delta", "improved"],
    )
    write_csv(
        os.path.join(OUT_ROOT, "results_summary_confirm.csv"),
        confirm_summary,
        ["task_name", "run_name", "dataset_profile", "seed", "use_pinyin", "best_cor_f1", "best_step", "max_train_steps", "run_dir"],
    )
    write_csv(
        os.path.join(OUT_ROOT, "ablation_table_confirm.csv"),
        confirm_ablation,
        ["dataset_profile", "use_pinyin", "count", "mean_best_cor_f1"],
    )
    write_csv(
        os.path.join(OUT_ROOT, "stability_table_confirm.csv"),
        confirm_stability,
        ["dataset_profile", "seed", "baseline_f1", "pinyin_f1", "delta", "improved"],
    )

    plot_curves(rows)
    print(
        "Generated: results_summary.csv, ablation_table.csv, stability_table.csv, "
        "results_summary_confirm.csv, ablation_table_confirm.csv, stability_table_confirm.csv, plots/*"
    )


if __name__ == "__main__":
    main()
