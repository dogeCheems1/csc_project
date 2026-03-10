#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage1 mid confirmation on SIGHAN2013/2014/2015."""

import json
import os
import sys
from glob import glob

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


def _make_key(dataset_profile, seed, use_pinyin):
    return f"{dataset_profile}|{seed}|{int(use_pinyin)}"


def _load_existing_results(out_path):
    if not os.path.exists(out_path):
        return []
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_done_keys_from_metrics():
    done = set()
    metrics_paths = glob(
        os.path.join(
            path_root,
            "macro_correct",
            "output",
            "text_correction",
            "stage1_confirm",
            "*",
            "metrics.json",
        )
    )
    for p in metrics_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            done.add(_make_key(m["dataset_profile"], int(m["seed"]), bool(m["use_pinyin"])))
        except Exception:
            continue
    return done


def _save_results(out_path, rows):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    datasets = ["sighan2013", "sighan2014", "sighan2015"]
    seeds = [42, 43, 44]
    out_path = os.path.join(path_root, "macro_correct", "output", "text_correction", "stage1_confirm_results.json")

    results = _load_existing_results(out_path)
    done_keys = {_make_key(r["dataset_profile"], int(r["seed"]), bool(r["use_pinyin"])) for r in results}
    done_keys.update(_load_done_keys_from_metrics())

    for dataset_profile in datasets:
        for seed in seeds:
            for use_pinyin in [False, True]:
                key = _make_key(dataset_profile, seed, use_pinyin)
                if key in done_keys:
                    print(f"[skip] {key}")
                    continue
                label = "pinyin" if use_pinyin else "baseline"
                metrics = train_csc(
                    {
                        "dataset_profile": dataset_profile,
                        "task_name": "stage1_confirm",
                        "run_name": f"confirm_{dataset_profile}_{label}",
                        "seed": seed,
                        "use_pinyin": use_pinyin,
                        "max_train_steps": 1200,
                        "save_steps": 150,
                        "early_stop_patience": 3,
                    }
                )
                results.append(metrics)
                done_keys.add(key)
                _save_results(out_path, results)
                print(f"[done] {key}")

    _save_results(out_path, results)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
