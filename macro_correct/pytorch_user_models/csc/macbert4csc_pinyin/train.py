# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Training pipeline for MacBERT + pinyin fusion CSC."""

from __future__ import absolute_import, division, print_function

import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc.dataset import (  # noqa: E402
    DataSetProcessor,
    save_json,
    sent_mertic_cor,
    sent_mertic_det,
)
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.config import (  # noqa: E402
    apply_dataset_profile,
    csc_config,
    to_namespace,
)
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.graph import (  # noqa: E402
    Macbert4CSCWithPinyin as Graph,
)


@dataclass
class InputFeatures:
    src_ids: List[int]
    attention_mask: List[int]
    trg_ids: List[int]
    pinyin_ids: List[List[int]]
    pinyin_lengths: List[int]


def build_run_dir(args):
    if not getattr(args, "run_name", None):
        args.run_name = args.task_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.model_save_path, args.task_name, f"{args.run_name}_seed{args.seed}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logger(run_dir):
    logger_name = f"macbert4csc_pinyin_{os.path.basename(run_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_random_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_args(overrides: Optional[Dict] = None):
    cfg = copy.deepcopy(vars(csc_config))
    if overrides:
        cfg.update(overrides)
    cfg = apply_dataset_profile(cfg)
    if cfg.get("task_name") is None:
        cfg["task_name"] = cfg.get("run_name", "default_task")
    return to_namespace(cfg)


def convert_examples_to_features(examples, max_seq_length, tokenizer, model, logger=None):
    features = []
    for i, example in tqdm(enumerate(examples), total=len(examples), desc="preprocess"):
        encoded_inputs = tokenizer(
            example.src,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True,
        )

        trg_ids = tokenizer(
            example.trg,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True,
        )["input_ids"]

        pinyin_ids, pinyin_lengths = model.text_to_pinyin_ids(example.src, encoded_inputs["input_ids"])

        if i < 3 and logger:
            logger.info(f"[sample] id={example.guid}")
            logger.info(f"[sample] src={''.join(example.src)}")
            logger.info(f"[sample] trg={''.join(example.trg)}")

        features.append(
            InputFeatures(
                src_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                trg_ids=trg_ids,
                pinyin_ids=pinyin_ids,
                pinyin_lengths=pinyin_lengths,
            )
        )
    return features


def features_to_dataloader(features, batch_size, sampler_cls):
    all_input_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.trg_ids for f in features], dtype=torch.long)
    all_pinyin_ids = torch.tensor([f.pinyin_ids for f in features], dtype=torch.long)
    all_pinyin_lengths = torch.tensor([f.pinyin_lengths for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_pinyin_ids, all_pinyin_lengths)
    sampler = sampler_cls(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)


def evaluate(model, eval_dataloader, tokenizer, device, logger):
    model.eval()
    eval_loss = 0.0
    eval_det_loss = 0.0
    eval_cor_loss = 0.0
    all_inputs, all_labels, all_predictions = [], [], []

    for eval_batch in tqdm(eval_dataloader, desc="eval"):
        eval_batch = tuple(t.to(device) for t in eval_batch)
        src_ids, attention_mask, trg_ids, pinyin_ids, pinyin_lengths = eval_batch
        with torch.no_grad():
            outputs = model(
                input_ids=src_ids,
                attention_mask=attention_mask,
                labels=trg_ids,
                pinyin_ids=pinyin_ids,
                pinyin_lengths=pinyin_lengths,
            )
        tmp_eval_loss = outputs[0]
        tmp_eval_det_loss = outputs[1]
        tmp_eval_cor_loss = outputs[2]
        logits = outputs[-2]
        eval_loss += tmp_eval_loss.mean().item()
        eval_det_loss += tmp_eval_det_loss.mean().item()
        eval_cor_loss += tmp_eval_cor_loss.mean().item()

        _, prd_ids = torch.max(logits, -1)
        prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
        src_ids = src_ids.cpu().numpy().tolist()
        trg_ids = trg_ids.cpu().numpy().tolist()
        for s, t, p in zip(src_ids, trg_ids, prd_ids):
            mapped_src = []
            mapped_trg = []
            mapped_prd = []
            for st, tt, pt in zip(s, t, p):
                if st == tokenizer.sep_token_id or st == tokenizer.cls_token_id:
                    continue
                mapped_src.append(st)
                mapped_trg.append(tt)
                mapped_prd.append(pt if st != pt else st)
            all_inputs.append(tokenizer.convert_ids_to_tokens(mapped_src, skip_special_tokens=True))
            all_labels.append(tokenizer.convert_ids_to_tokens(mapped_trg, skip_special_tokens=True))
            all_predictions.append(tokenizer.convert_ids_to_tokens(mapped_prd, skip_special_tokens=True))

    det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(all_inputs, all_predictions, all_labels, logger)
    cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(all_inputs, all_predictions, all_labels, logger)
    size = max(1, len(eval_dataloader))
    stats = {
        "eval_loss": eval_loss / size,
        "eval_det_loss": eval_det_loss / size,
        "eval_cor_loss": eval_cor_loss / size,
        "det_acc": det_acc,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,
        "cor_acc": cor_acc,
        "cor_precision": cor_precision,
        "cor_recall": cor_recall,
        "cor_f1": cor_f1,
    }

    if all_inputs:
        logger.info("[eval] sample prediction:")
        logger.info(f"[eval] src: {''.join(all_inputs[0])}")
        logger.info(f"[eval] trg: {''.join(all_labels[0])}")
        logger.info(f"[eval] prd: {''.join(all_predictions[0])}")

    sample_records = []
    for src, trg, prd in zip(all_inputs, all_labels, all_predictions):
        src_s = "".join(src)
        trg_s = "".join(trg)
        prd_s = "".join(prd)
        sample_records.append(
            {
                "src": src_s,
                "trg": trg_s,
                "pred": prd_s,
                "changed": src_s != prd_s,
                "corrected": prd_s == trg_s,
            }
        )
    success = [x for x in sample_records if x["corrected"]]
    failure = [x for x in sample_records if not x["corrected"]]
    stats["sample_analysis"] = (success[:10] + failure[:10])[:20]

    gate_stats = getattr(model, "last_gate_stats", None)
    if isinstance(gate_stats, dict):
        stats["gate_mean"] = gate_stats.get("mean")
        stats["gate_std"] = gate_stats.get("std")

    return stats


def train_csc(config_overrides: Optional[Dict] = None):
    args = get_args(config_overrides)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES or "0"
    os.environ["USE_TORCH"] = args.USE_TORCH or "1"
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    run_dir = build_run_dir(args)
    logger = setup_logger(run_dir)
    tb_writer = SummaryWriter(logdir=os.path.join(run_dir, "tb"))
    logger.info(f"run_dir={run_dir}")
    logger.info(f"dataset_profile={args.dataset_profile}, use_pinyin={args.use_pinyin}, seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.flag_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"device={device}, n_gpu={n_gpu}")
    set_random_seed(args.seed, n_gpu)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            do_lower_case=args.do_lower_case,
            use_fast=args.flag_fast_tokenizer,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            do_lower_case=args.do_lower_case,
            use_fast=args.flag_fast_tokenizer,
        )

    processor = DataSetProcessor(
        path_train=args.path_train,
        path_dev=args.path_dev,
        path_tet=args.path_tet,
        task_name=args.task_name,
    )

    model = Graph(config=args, csc_config=args)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_examples = processor.get_train_examples()
    eval_examples = processor.get_dev_examples()
    if getattr(args, "max_train_samples", None):
        max_train_samples = int(args.max_train_samples)
        if max_train_samples > 0 and len(train_examples) > max_train_samples:
            rng = random.Random(args.seed)
            indices = rng.sample(range(len(train_examples)), max_train_samples)
            train_examples = [train_examples[i] for i in indices]
            logger.info(f"train_examples downsampled to {len(train_examples)} (max_train_samples={max_train_samples})")
    logger.info(f"train_examples={len(train_examples)}, dev_examples={len(eval_examples)}")

    model_for_feature = model.module if hasattr(model, "module") else model
    train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, model_for_feature, logger)
    eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, model_for_feature, logger)
    train_dataloader = features_to_dataloader(train_features, args.train_batch_size, RandomSampler)
    eval_dataloader = features_to_dataloader(eval_features, args.eval_batch_size, SequentialSampler)

    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    if args.max_train_steps is None:
        args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps else int(args.max_train_steps * args.warmup_proportion)
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=args.lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    scaler = None
    if args.flag_fp16:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()

    logger.info("***** start training *****")
    logger.info(f"max_train_steps={args.max_train_steps}, num_train_epochs={args.num_train_epochs}, save_steps={args.save_steps}")

    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0
    best_f1 = -1.0
    best_step = -1
    best_checkpoint = None
    not_improved_rounds = 0
    history = []
    best_sample_analysis = []

    for epoch in range(int(args.num_train_epochs)):
        logger.info(f"========== epoch {epoch + 1}/{args.num_train_epochs} ==========")
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, pinyin_ids, pinyin_lengths = batch
            outputs = model(
                input_ids=src_ids,
                attention_mask=attention_mask,
                labels=trg_ids,
                pinyin_ids=pinyin_ids,
                pinyin_lengths=pinyin_lengths,
            )
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.flag_fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.flag_fp16 and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                progress_bar.update(1)

                if global_step % args.save_steps == 0:
                    stats = evaluate(model, eval_dataloader, tokenizer, device, logger)
                    stats["global_step"] = global_step
                    history.append(stats)

                    logger.info(
                        "[eval] step=%s cor_f1=%.4f det_f1=%.4f cor_p=%.4f cor_r=%.4f",
                        global_step,
                        stats["cor_f1"],
                        stats["det_f1"],
                        stats["cor_precision"],
                        stats["cor_recall"],
                    )
                    if "gate_mean" in stats and stats["gate_mean"] is not None:
                        logger.info("[gate] mean=%.4f std=%.4f", stats["gate_mean"], stats["gate_std"])

                    tb_writer.add_scalar("eval/cor_f1", stats["cor_f1"], global_step)
                    tb_writer.add_scalar("eval/det_f1", stats["det_f1"], global_step)
                    tb_writer.add_scalar("eval/loss", stats["eval_loss"], global_step)
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                    if stats["cor_f1"] > best_f1:
                        best_f1 = stats["cor_f1"]
                        best_step = global_step
                        best_sample_analysis = stats.get("sample_analysis", [])
                        not_improved_rounds = 0
                        model_to_save = model.module if hasattr(model, "module") else model
                        best_checkpoint = os.path.join(run_dir, f"best_model_step-{global_step}_f1-{best_f1:.4f}.bin")
                        torch.save(model_to_save.state_dict(), best_checkpoint)
                        model_to_save.config.save_pretrained(save_directory=run_dir)
                        tokenizer.save_pretrained(save_directory=run_dir)
                    else:
                        not_improved_rounds += 1
                        if not_improved_rounds >= args.early_stop_patience:
                            logger.info("early stop triggered at step=%s", global_step)
                            break

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps or not_improved_rounds >= args.early_stop_patience:
            break

    metrics = {
        "run_name": args.run_name,
        "task_name": args.task_name,
        "dataset_profile": args.dataset_profile,
        "seed": args.seed,
        "use_pinyin": args.use_pinyin,
        "fusion_type": args.fusion_type,
        "max_train_steps": args.max_train_steps,
        "save_steps": args.save_steps,
        "best_step": best_step,
        "best_cor_f1": float(best_f1 if best_f1 >= 0 else 0.0),
        "best_checkpoint": best_checkpoint,
        "history": history,
        "run_dir": run_dir,
        "sample_analysis": best_sample_analysis,
    }

    save_json(vars(args), os.path.join(run_dir, "csc.config"), mode="w")
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "sample_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(best_sample_analysis, f, ensure_ascii=False, indent=2)

    logger.info("training finished: best_cor_f1=%.4f at step=%s", metrics["best_cor_f1"], best_step)
    tb_writer.close()
    return metrics


if __name__ == "__main__":
    train_csc()
