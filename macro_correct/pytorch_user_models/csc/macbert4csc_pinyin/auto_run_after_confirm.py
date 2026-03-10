#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto pipeline after Stage1 confirm is complete.

Flow:
1) Wait until stage1_confirm has all expected 18 runs.
2) Run collect_results.py
3) Run one long final pinyin run on SIGHAN2015
4) Run stage2_wang.py
5) Run collect_results.py again
"""

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


OUT_ROOT = os.path.join(path_root, "macro_correct", "output", "text_correction")
STAGE1_DIR = os.path.join(OUT_ROOT, "stage1_confirm")

EXPECTED_DATASETS = ["sighan2013", "sighan2014", "sighan2015"]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_PINYIN = [False, True]
EXPECTED_TOTAL = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS) * len(EXPECTED_PINYIN)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_stage1_confirm_metrics():
    rows = []
    metrics_paths = glob.glob(os.path.join(STAGE1_DIR, "*", "metrics.json"))
    for p in metrics_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            if m.get("task_name") != "stage1_confirm":
                continue
            key = (m.get("dataset_profile"), int(m.get("seed")), bool(m.get("use_pinyin")))
            rows.append((key, m))
        except Exception:
            continue
    return rows


def wait_stage1_confirm_done(check_interval_sec=120):
    while True:
        rows = load_stage1_confirm_metrics()
        uniq = {k for k, _ in rows}
        print(f"[{now()}] stage1_confirm done={len(uniq)}/{EXPECTED_TOTAL}")
        if len(uniq) >= EXPECTED_TOTAL:
            return [m for _, m in rows]
        time.sleep(check_interval_sec)


def run_cmd(cmd):
    print(f"[{now()}] run: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=path_root, check=True)


def choose_best_seed_from_confirm(metrics):
    cand = [
        m
        for m in metrics
        if m.get("dataset_profile") == "sighan2015" and bool(m.get("use_pinyin")) is True
    ]
    if not cand:
        return 42
    cand.sort(key=lambda x: float(x.get("best_cor_f1", 0.0)), reverse=True)
    return int(cand[0].get("seed", 42))


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 1) Wait stage1 confirm complete
    metrics = wait_stage1_confirm_done()

    # 2) Collect current results
    run_cmd([sys.executable, "macro_correct/pytorch_user_models/csc/macbert4csc_pinyin/collect_results.py"])

    # 3) One long final pinyin run on SIGHAN2015
    seed = choose_best_seed_from_confirm(metrics)
    print(f"[{now()}] chosen seed for stage1_long_final: {seed}")
    long_metrics = train_csc(
        {
            "dataset_profile": "sighan2015",
            "task_name": "stage1_long_final",
            "run_name": "final_pinyin_long",
            "seed": seed,
            "use_pinyin": True,
            "max_train_steps": 2800,
            "save_steps": 200,
            "early_stop_patience": 3,
        }
    )
    with open(os.path.join(OUT_ROOT, "stage1_long_final_result.json"), "w", encoding="utf-8") as f:
        json.dump(long_metrics, f, ensure_ascii=False, indent=2)

    # 4) Stage2 extension
    run_cmd([sys.executable, "macro_correct/pytorch_user_models/csc/macbert4csc_pinyin/run_stage2_wang.py"])

    # 5) Final collect
    run_cmd([sys.executable, "macro_correct/pytorch_user_models/csc/macbert4csc_pinyin/collect_results.py"])
    print(f"[{now()}] auto pipeline finished.")


if __name__ == "__main__":
    main()

