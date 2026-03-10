#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage1 short screening: baseline vs pinyin on SIGHAN2015."""

import json
import os
import sys

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


def main():
    seeds = [42, 43, 44]
    results = []
    for seed in seeds:
        for use_pinyin in [False, True]:
            label = "pinyin" if use_pinyin else "baseline"
            metrics = train_csc(
                {
                    "dataset_profile": "sighan2015",
                    "task_name": "stage1_screen",
                    "run_name": f"screen_{label}",
                    "seed": seed,
                    "use_pinyin": use_pinyin,
                    "max_train_steps": 400,
                    "save_steps": 100,
                }
            )
            results.append(metrics)

    out_path = os.path.join(path_root, "macro_correct", "output", "text_correction", "stage1_screen_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
