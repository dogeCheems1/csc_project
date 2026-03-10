#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one long baseline (no pinyin) experiment on SIGHAN2015."""

import json
import os
import sys

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


OUT_ROOT = os.path.join(path_root, "macro_correct", "output", "text_correction")
OUT_JSON = os.path.join(OUT_ROOT, "stage1_long_baseline_result.json")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    metrics = train_csc(
        {
            "dataset_profile": "sighan2015",
            "task_name": "stage1_long_baseline",
            "run_name": "final_baseline_long",
            "seed": 44,
            "use_pinyin": False,
            "max_train_steps": 2800,
            "save_steps": 200,
            "early_stop_patience": 3,
        }
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

