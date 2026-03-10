# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Runtime config for MacBERT + pinyin fusion CSC experiments."""

from argparse import Namespace
import os
import platform


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CORPUS_ROOT = os.path.join(PROJECT_ROOT, "macro_correct", "corpus", "text_correction")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "macro_correct", "output", "text_correction")


SIGHAN_PATHS = {
    "sighan2013": {
        "train": os.path.join(CORPUS_ROOT, "sighan", "sighan2013.train.json"),
        "dev": os.path.join(CORPUS_ROOT, "sighan", "sighan2013.dev.json"),
        "tet": os.path.join(CORPUS_ROOT, "sighan", "sighan2013.dev.json"),
    },
    "sighan2014": {
        "train": os.path.join(CORPUS_ROOT, "sighan", "sighan2014.train.json"),
        "dev": os.path.join(CORPUS_ROOT, "sighan", "sighan2014.dev.json"),
        "tet": os.path.join(CORPUS_ROOT, "sighan", "sighan2014.dev.json"),
    },
    "sighan2015": {
        "train": os.path.join(CORPUS_ROOT, "sighan", "sighan2015.train.json"),
        "dev": os.path.join(CORPUS_ROOT, "sighan", "sighan2015.dev.json"),
        "tet": os.path.join(CORPUS_ROOT, "sighan", "sighan2015.dev.json"),
    },
}


WANG271K_PATHS = {
    "train": os.path.join(CORPUS_ROOT, "wang271k", "train.json"),
    "dev": os.path.join(CORPUS_ROOT, "wang271k", "valid.json"),
    "tet": os.path.join(CORPUS_ROOT, "wang271k", "test.json"),
}


def _resolve_profile(profile: str):
    if profile in SIGHAN_PATHS:
        return SIGHAN_PATHS[profile]

    if profile == "sighan_all":
        return {
            "train": [SIGHAN_PATHS["sighan2013"]["train"], SIGHAN_PATHS["sighan2014"]["train"], SIGHAN_PATHS["sighan2015"]["train"]],
            "dev": [SIGHAN_PATHS["sighan2013"]["dev"], SIGHAN_PATHS["sighan2014"]["dev"], SIGHAN_PATHS["sighan2015"]["dev"]],
            "tet": [SIGHAN_PATHS["sighan2013"]["tet"], SIGHAN_PATHS["sighan2014"]["tet"], SIGHAN_PATHS["sighan2015"]["tet"]],
        }

    if profile == "sighan_plus_wang":
        train_paths = [SIGHAN_PATHS["sighan2013"]["train"], SIGHAN_PATHS["sighan2014"]["train"], SIGHAN_PATHS["sighan2015"]["train"]]
        if os.path.exists(WANG271K_PATHS["train"]):
            train_paths.append(WANG271K_PATHS["train"])
        return {
            "train": train_paths,
            "dev": SIGHAN_PATHS["sighan2015"]["dev"],
            "tet": SIGHAN_PATHS["sighan2015"]["tet"],
        }

    raise ValueError(f"Unsupported dataset_profile: {profile}")


def _default_config():
    is_windows = platform.system().lower() == "windows"
    return {
        "pretrained_model_name_or_path": "hfl/chinese-macbert-base",
        "dataset_profile": "sighan2015",
        "run_name": "default_run",
        "task_name": "default_run",
        "model_save_path": OUTPUT_ROOT,
        "path_train": SIGHAN_PATHS["sighan2015"]["train"],
        "path_dev": SIGHAN_PATHS["sighan2015"]["dev"],
        "path_tet": SIGHAN_PATHS["sighan2015"]["tet"],
        "do_lower_case": True,
        "do_train": True,
        "do_eval": True,
        "do_test": True,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "learning_rate": 3e-5,
        "max_train_steps": 300,
        "max_train_steps_short": 300,
        "max_train_steps_mid": 1500,
        "max_train_steps_long": 2800,
        "num_train_epochs": None,
        "warmup_proportion": 0.1,
        "num_warmup_steps": None,
        "max_seq_length": 128,
        "max_train_samples": None,
        "max_grad_norm": 1.0,
        "weight_decay": 5e-4,
        "save_steps": 100,
        "seed": 42,
        "seed_list": [42, 43, 44],
        "lr_scheduler_type": "linear",
        "loss_type": "BCE",
        "loss_det_rate": 0.3,
        "use_pinyin": True,
        "fusion_type": "gate",
        "pinyin_vocab_size": 500,
        "pinyin_embed_dim": 128,
        "early_stop_patience": 3,
        "eval_key": "cor_f1",
        "flag_fast_tokenizer": True,
        "flag_train": True,
        "flag_fp16": not is_windows,
        "flag_cuda": True,
        "flag_skip": True,
        "num_workers": 0 if is_windows else 4,
        "CUDA_VISIBLE_DEVICES": "0",
        "USE_TORCH": "1",
        "smoke_steps": 50,
    }


def apply_dataset_profile(config_dict):
    paths = _resolve_profile(config_dict["dataset_profile"])
    config_dict["path_train"] = paths["train"]
    config_dict["path_dev"] = paths["dev"]
    config_dict["path_tet"] = paths["tet"]
    return config_dict


def to_namespace(config_dict):
    return Namespace(**config_dict)


csc_config = to_namespace(apply_dataset_profile(_default_config()))
