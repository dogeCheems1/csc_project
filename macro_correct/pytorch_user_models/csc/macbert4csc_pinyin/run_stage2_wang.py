#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage2 extension: compare with/without Wang271k."""

import json
import os
import sys
from glob import glob
from typing import Dict, List

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


def _make_key(run_name, seed):
    return f"{run_name}|{seed}"


def _load_done_keys():
    done = set()
    metrics_paths = glob(
        os.path.join(
            path_root,
            "macro_correct",
            "output",
            "text_correction",
            "stage2_wang",
            "*",
            "metrics.json",
        )
    )
    for p in metrics_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            done.add(_make_key(m["run_name"], int(m["seed"])))
        except Exception:
            continue
    return done


def _run_with_oom_fallback(cfg: Dict, sample_plan: List[int]):
    """Retry stage2 run with smaller sampled training data if CPU OOM occurs."""
    if cfg.get("max_train_samples") is None:
        return train_csc(cfg)

    last_error = None
    for n in sample_plan:
        trial_cfg = dict(cfg)
        trial_cfg["max_train_samples"] = n
        print(f"[run] {trial_cfg['run_name']}|{trial_cfg['seed']} max_train_samples={n}")
        try:
            return train_csc(trial_cfg)
        except RuntimeError as e:
            msg = str(e).lower()
            if "not enough memory" in msg or "defaultcpuallocator" in msg:
                last_error = e
                print(f"[oom] retry with smaller subset, reason={e}")
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("stage2 run failed before starting retry logic")


def main():
    configs = [
        {
            "dataset_profile": "sighan_all",
            "task_name": "stage2_wang",
            "run_name": "without_wang",
            "seed": 42,
            "use_pinyin": True,
            "max_train_steps": 1800,
            "save_steps": 150,
        },
        {
            "dataset_profile": "sighan_plus_wang",
            "task_name": "stage2_wang",
            "run_name": "with_wang",
            "seed": 42,
            "use_pinyin": True,
            "max_train_samples": 40000,
            "max_train_steps": 1800,
            "save_steps": 150,
        },
    ]
    done = _load_done_keys()
    results = []
    for cfg in configs:
        key = _make_key(cfg["run_name"], cfg["seed"])
        if key in done:
            print(f"[skip] {key}")
            continue
        if cfg["run_name"] == "with_wang":
            results.append(_run_with_oom_fallback(cfg, sample_plan=[40000, 30000, 20000, 12000]))
        else:
            results.append(train_csc(cfg))
    out_path = os.path.join(path_root, "macro_correct", "output", "text_correction", "stage2_wang_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
