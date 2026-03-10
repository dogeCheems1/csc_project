#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare final artifacts for thesis delivery."""

import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from glob import glob
from statistics import mean


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
OUT_ROOT = os.path.join(ROOT, "macro_correct", "output", "text_correction")
FINAL_DIR = os.path.join(OUT_ROOT, "final_artifacts")
PLOTS_DIR = os.path.join(FINAL_DIR, "plots_confirm_stage2")


def load_metrics_rows():
    rows = []
    for path in glob(os.path.join(OUT_ROOT, "**", "metrics.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            rows.append(m)
        except Exception:
            continue
    return rows


def safe_copy(src, dst_dir):
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    return False


def score_readability(text):
    if not text:
        return -1.0
    commons = "的一是不我人在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现"
    good = sum(1 for ch in text if ch in commons)
    bad_tokens = ["�", "锟", "\x00"]
    bad = sum(text.count(t) for t in bad_tokens)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ratio = cjk / max(1, len(text))
    return good * 2 + ratio * 5 - bad * 10


def try_fix_mojibake(text):
    if not text:
        return text
    cands = {text}
    for enc in ("gbk", "gb18030", "big5", "latin1"):
        try:
            cands.add(text.encode(enc, errors="ignore").decode("utf-8", errors="ignore"))
        except Exception:
            pass
    best = max(cands, key=score_readability)
    return best.strip()


def normalize_record(rec):
    src = try_fix_mojibake(rec.get("src", ""))
    trg = try_fix_mojibake(rec.get("trg", ""))
    pred = try_fix_mojibake(rec.get("pred", ""))
    return {
        "src": src,
        "trg": trg,
        "pred": pred,
        "changed": bool(rec.get("changed", src != pred)),
        "corrected": bool(rec.get("corrected", pred == trg)),
    }


def build_sample_analysis(rows):
    allowed_tasks = {"stage1_confirm", "stage2_wang", "stage1_long_final", "stage1_long_baseline"}
    pool = []
    for m in rows:
        if m.get("task_name") not in allowed_tasks:
            continue
        if str(m.get("run_name", "")).startswith("smoke"):
            continue
        hist = m.get("history", [])
        for h in hist:
            for rec in h.get("sample_analysis", []):
                nrec = normalize_record(rec)
                text = nrec["src"] + nrec["trg"] + nrec["pred"]
                if len(text) < 12:
                    continue
                # Drop obviously broken predictions from low-quality early runs.
                if "##" in nrec["pred"] or "eeworld" in nrec["pred"] or "mobil" in nrec["pred"]:
                    continue
                if score_readability(nrec["pred"]) < 1.0:
                    continue
                pool.append(
                    {
                        "task_name": m.get("task_name"),
                        "run_name": m.get("run_name"),
                        "dataset_profile": m.get("dataset_profile"),
                        **nrec,
                    }
                )

    uniq = {}
    for item in pool:
        key = (item["src"], item["trg"], item["pred"], item["corrected"])
        if key not in uniq:
            uniq[key] = item
    items = list(uniq.values())
    success = [x for x in items if x["corrected"]]
    failure = [x for x in items if not x["corrected"]]
    # Prefer readable records with actual correction difficulty.
    failure = [x for x in failure if x["src"] != x["trg"]]
    return success[:10], failure[:10]


def save_samples(success, failure):
    out_json = os.path.join(FINAL_DIR, "sample_analysis_20.json")
    out_md = os.path.join(FINAL_DIR, "sample_analysis_20.md")
    payload = {"success_10": success, "failure_10": failure}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["# Sample Analysis (20)", "", "## Success (10)"]
    for i, s in enumerate(success, 1):
        lines.extend(
            [
                f"{i}. [{s['dataset_profile']}|{s['run_name']}]",
                f"   src: {s['src']}",
                f"   trg: {s['trg']}",
                f"   pred: {s['pred']}",
            ]
        )
    lines.append("")
    lines.append("## Failure (10)")
    for i, s in enumerate(failure, 1):
        lines.extend(
            [
                f"{i}. [{s['dataset_profile']}|{s['run_name']}]",
                f"   src: {s['src']}",
                f"   trg: {s['trg']}",
                f"   pred: {s['pred']}",
            ]
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_confirm_stage2(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    filt = [r for r in rows if r.get("task_name") in {"stage1_confirm", "stage2_wang"}]

    # curve
    plt.figure(figsize=(10, 6))
    for r in filt:
        hist = r.get("history", [])
        if not hist:
            continue
        xs = [h.get("global_step") for h in hist if h.get("global_step") is not None]
        ys = [h.get("cor_f1") for h in hist if h.get("cor_f1") is not None]
        if not xs or not ys:
            continue
        label = f"{r.get('task_name')}|{r.get('dataset_profile')}|seed{r.get('seed')}|py{int(bool(r.get('use_pinyin')))}"
        plt.plot(xs, ys, label=label, alpha=0.75)
    plt.xlabel("global_step")
    plt.ylabel("cor_f1")
    plt.title("F1-step Curves (stage1_confirm + stage2_wang)")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "f1_step_curves_confirm_stage2.png"), dpi=150)
    plt.close()

    # bar
    grouped = defaultdict(lambda: {False: [], True: []})
    for r in filt:
        grouped[r.get("dataset_profile")][bool(r.get("use_pinyin"))].append(float(r.get("best_cor_f1", 0.0)))
    ds = sorted(grouped.keys())
    base = [mean(grouped[d][False]) if grouped[d][False] else 0.0 for d in ds]
    py = [mean(grouped[d][True]) if grouped[d][True] else 0.0 for d in ds]
    x = range(len(ds))
    w = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - w / 2 for i in x], base, width=w, label="baseline")
    plt.bar([i + w / 2 for i in x], py, width=w, label="pinyin")
    plt.xticks(list(x), ds, rotation=20)
    plt.ylabel("mean best_cor_f1")
    plt.title("Dataset Comparison (stage1_confirm + stage2_wang)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_bar_confirm_stage2.png"), dpi=150)
    plt.close()


def write_manifest(copied):
    path = os.path.join(FINAL_DIR, "manifest.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "copied_files": copied,
        "plots_dir": PLOTS_DIR,
        "sample_files": [
            os.path.join(FINAL_DIR, "sample_analysis_20.json"),
            os.path.join(FINAL_DIR, "sample_analysis_20.md"),
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)
    copied = []
    key_files = [
        os.path.join(OUT_ROOT, "results_summary_confirm.csv"),
        os.path.join(OUT_ROOT, "ablation_table_confirm.csv"),
        os.path.join(OUT_ROOT, "stability_table_confirm.csv"),
        os.path.join(OUT_ROOT, "results_summary.csv"),
        os.path.join(OUT_ROOT, "ablation_table.csv"),
        os.path.join(OUT_ROOT, "stability_table.csv"),
        os.path.join(OUT_ROOT, "stage1_long_final_result.json"),
        os.path.join(OUT_ROOT, "stage1_long_baseline_result.json"),
        os.path.join(OUT_ROOT, "stage2_wang_results.json"),
    ]
    for p in key_files:
        if safe_copy(p, FINAL_DIR):
            copied.append(p)

    rows = load_metrics_rows()
    plot_confirm_stage2(rows)
    success, failure = build_sample_analysis(rows)
    save_samples(success, failure)
    write_manifest(copied)
    print(f"prepared: {FINAL_DIR}")
    print(f"samples: success={len(success)}, failure={len(failure)}")


if __name__ == "__main__":
    main()
