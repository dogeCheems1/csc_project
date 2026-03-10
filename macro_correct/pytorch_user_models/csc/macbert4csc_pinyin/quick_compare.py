#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick baseline vs pinyin comparison with isolated run outputs."""

import json
import os
import sys

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


def quick_compare():
    runs = []
    common = {
        "dataset_profile": "sighan2015",
        "task_name": "stage1_quick_compare",
        "max_train_steps": 300,
        "save_steps": 100,
        "seed": 42,
        "run_name": "quick",
    }
    for use_pinyin in [False, True]:
        run_name = "quick_pinyin" if use_pinyin else "quick_baseline"
        metrics = train_csc(
            {
                **common,
                "use_pinyin": use_pinyin,
                "run_name": run_name,
            }
        )
        runs.append(metrics)

    summary_path = os.path.join(path_root, "macro_correct", "output", "text_correction", "quick_compare_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    print(f"quick compare done. summary: {summary_path}")
    if len(runs) == 2:
        delta = runs[1]["best_cor_f1"] - runs[0]["best_cor_f1"]
        print(f"baseline={runs[0]['best_cor_f1']:.4f}, pinyin={runs[1]['best_cor_f1']:.4f}, delta={delta:+.4f}")


if __name__ == "__main__":
    quick_compare()




