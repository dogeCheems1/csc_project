# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
训练脚本：基于 MacBERT 与拼音特征融合的中文拼写纠错模型
作者：[你的名字]
毕业设计：《基于 MacBERT 与拼音特征融合的中文拼写错误检测与纠正系统设计与实现》
"""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import random
import math
import sys
import os

# 添加项目根目录到路径
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

# 导入配置（稍后创建）
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.config import csc_config as args

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES or "0"
os.environ["USE_TORCH"] = args.USE_TORCH or "1"

# ⭐ 关键：清除代理设置（避免与本地代理冲突）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 设置 HuggingFace 为离线模式（优先使用本地缓存）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_scheduler
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

# 导入数据处理和模型
from macro_correct.pytorch_user_models.csc.macbert4csc.dataset import (
    DataSetProcessor, sent_mertic_det, sent_mertic_cor, save_json
)
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.graph import Macbert4CSCWithPinyin as Graph


class InputFeatures(object):
    """
    数据特征类：存储模型输入所需的所有特征
    
    【论文对应】第3章 系统设计 - 3.2 数据预处理
    """
    def __init__(self, src_ids, attention_mask, trg_ids, pinyin_ids, pinyin_lengths):
        self.src_ids = src_ids              # 输入文本的 token ID
        self.attention_mask = attention_mask  # 注意力掩码（区分真实token和padding）
        self.trg_ids = trg_ids              # 目标文本的 token ID（正确答案）
        self.pinyin_ids = pinyin_ids        # ⭐ 拼音特征ID（核心创新）
        self.pinyin_lengths = pinyin_lengths  # ⭐ 拼音实际长度（用于Mask）


def convert_examples_to_features(examples, max_seq_length, tokenizer, model, logger=None):
    """
    将原始样本转换为模型输入特征
    
    【论文对应】第3章 系统设计 - 3.2.2 拼音特征提取
    
    【为什么要做】
    模型不能直接处理文本，需要转换为：
    1. Token ID（BERT 输入）
    2. 拼音 ID（拼音编码器输入）⭐ 核心创新
    3. Attention Mask（区分真实内容和填充）
    """
    features = []
    for i, example in tqdm(enumerate(examples), desc="数据预处理"):
        # 1. 使用 BERT Tokenizer 编码文本
        encoded_inputs = tokenizer(
            example.src,  # 错误的句子
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True  # 输入已经是字符列表
        )
        
        # 2. 编码目标文本（正确答案）
        trg_ids = tokenizer(
            example.trg,  # 正确的句子
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True
        )["input_ids"]
        
        # 3. ⭐ 核心创新：提取拼音特征
        # 将文本转换为拼音ID（这是论文的核心贡献）
        text = "".join(example.src)  # 将字符列表合并为字符串
        pinyin_ids, pinyin_lengths = model.text_to_pinyin_ids([text], max_seq_length)
        pinyin_ids = pinyin_ids[0]  # 取第一个样本
        pinyin_lengths = pinyin_lengths[0]
        
        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        
        # 验证维度
        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(trg_ids) == max_seq_length
        assert pinyin_ids.shape[0] == max_seq_length
        
        # 打印前5个样本（用于调试）
        if i < 5 and logger:
            logger.info("*** 样本示例 ***")
            logger.info(f"ID: {example.guid}")
            logger.info(f"错误句子: {''.join(example.src)}")
            logger.info(f"正确句子: {''.join(example.trg)}")
            logger.info(f"拼音形状: {pinyin_ids.shape}")
        
        features.append(
            InputFeatures(
                src_ids=src_ids,
                attention_mask=attention_mask,
                trg_ids=trg_ids,
                pinyin_ids=pinyin_ids.tolist(),  # 转为列表存储
                pinyin_lengths=pinyin_lengths.tolist()
            )
        )
    return features


def train_csc():
    """
    主训练函数
    
    【论文对应】第4章 实验与分析 - 4.2 模型训练
    """
    # ========== 第1步：初始化配置 ==========
    processor = DataSetProcessor()
    processor.path_train = args.path_train
    processor.task_name = args.task_name
    processor.path_dev = args.path_dev
    processor.path_tet = args.path_tet
    task_name = args.task_name
    
    # 创建模型保存目录
    args.model_save_path = os.path.join(args.model_save_path, task_name)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    
    # 初始化 TensorBoard（用于可视化训练过程）
    tensorboardx_witer = SummaryWriter(logdir=args.model_save_path)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.model_save_path, "train.log")
    )
    logger = logging.getLogger(__name__)
    
    # 设置设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.flag_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"设备: {device}, GPU数量: {n_gpu}")
    
    # 设置随机种子（保证实验可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # ========== 第2步：加载 Tokenizer ==========
    logger.info("加载 Tokenizer...")
    try:
        # 优先从本地缓存加载（避免网络问题）
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            do_lower_case=args.do_lower_case,
            use_fast=args.flag_fast_tokenizer,
            local_files_only=True  # 只使用本地缓存
        )
    except Exception as e:
        logger.warning(f"从本地缓存加载失败，尝试从网络下载: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            do_lower_case=args.do_lower_case,
            use_fast=args.flag_fast_tokenizer
        )
    
    # ========== 第3步：初始化模型 ==========
    logger.info("初始化拼音融合模型...")
    model = Graph(config=args, csc_config=args)
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # ========== 第4步：准备训练数据 ==========
    if args.do_train:
        logger.info("加载训练数据...")
        train_examples = processor.get_train_examples()
        logger.info(f"训练样本数: {len(train_examples)}")
        
        # ⭐ 核心：将样本转换为特征（包含拼音）
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, 
            model.module if hasattr(model, "module") else model, 
            logger
        )
        
        # 转换为 Tensor
        all_input_ids = torch.tensor([f.src_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in train_features], dtype=torch.long)
        all_pinyin_ids = torch.tensor([f.pinyin_ids for f in train_features], dtype=torch.long)
        all_pinyin_lengths = torch.tensor([f.pinyin_lengths for f in train_features], dtype=torch.long)
        
        # 创建 DataLoader
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_label_ids, 
            all_pinyin_ids, all_pinyin_lengths
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )
        
        # 计算训练步数
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        logger.info(f"总训练步数: {args.max_train_steps}")
        logger.info(f"训练轮数: {args.num_train_epochs}")
    
    # ========== 第5步：配置优化器和学习率调度器 ==========
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # 学习率调度器（warmup + 线性衰减）
    num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps \
        else int(num_update_steps_per_epoch * args.warmup_proportion)
    scheduler = get_scheduler(
        optimizer=optimizer, 
        name=args.lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # 混合精度训练（节省显存）
    scaler = None
    if args.flag_fp16:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    
    # ========== 第6步：准备验证数据 ==========
    if args.do_eval:
        logger.info("加载验证数据...")
        eval_examples = processor.get_dev_examples()
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer,
            model.module if hasattr(model, "module") else model,
            logger
        )
        
        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
        all_pinyin_ids = torch.tensor([f.pinyin_ids for f in eval_features], dtype=torch.long)
        all_pinyin_lengths = torch.tensor([f.pinyin_lengths for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_label_ids,
            all_pinyin_ids, all_pinyin_lengths
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )
        logger.info(f"验证样本数: {len(eval_examples)}")
    
    # ========== 第7步：开始训练 ==========
    logger.info("***** 开始训练 *****")
    logger.info(f"  训练样本数 = {len(train_examples)}")
    logger.info(f"  Batch Size = {args.train_batch_size}")
    logger.info(f"  训练步数 = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0
    best_f1 = 0.0  # 记录最佳 F1 分数
    
    for epoch in range(int(args.num_train_epochs)):
        logger.info(f"\n========== Epoch {epoch+1}/{args.num_train_epochs} ==========")
        train_loss = 0
        train_det_loss = 0
        train_cor_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, pinyin_ids, pinyin_lengths = batch
            
            # ⭐ 核心：前向传播（包含拼音特征）
            if args.flag_fp16:
                with autocast():
                    outputs = model(
                        input_ids=src_ids,
                        attention_mask=attention_mask,
                        labels=trg_ids,
                        pinyin_ids=pinyin_ids,          # ⭐ 拼音ID
                        pinyin_lengths=pinyin_lengths    # ⭐ 拼音长度
                    )
            else:
                outputs = model(
                    input_ids=src_ids,
                    attention_mask=attention_mask,
                    labels=trg_ids,
                    pinyin_ids=pinyin_ids,
                    pinyin_lengths=pinyin_lengths
                )
            
            loss = outputs[0]
            tmp_det_loss = outputs[1]
            tmp_cor_loss = outputs[2]
            
            # 梯度累积
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            if args.flag_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item()
            train_det_loss += tmp_det_loss.mean().item()
            train_cor_loss += tmp_cor_loss.mean().item()
            
            # 更新参数
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.flag_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress_bar.update(1)
                global_step += 1
            
            # ========== 第8步：定期验证 ==========
            if args.do_eval and global_step % args.save_steps == 0 and \
               (step + 1) % args.gradient_accumulation_steps == 0:
                logger.info("\n***** 开始验证 *****")
                model.eval()
                eval_loss = 0
                eval_det_loss = 0
                eval_cor_loss = 0
                all_inputs, all_labels, all_predictions = [], [], []
                
                for eval_batch in tqdm(eval_dataloader, desc="验证中"):
                    eval_batch = tuple(t.to(device) for t in eval_batch)
                    src_ids, attention_mask, trg_ids, pinyin_ids, pinyin_lengths = eval_batch
                    
                    with torch.no_grad():
                        outputs = model(
                            input_ids=src_ids,
                            attention_mask=attention_mask,
                            labels=trg_ids,
                            pinyin_ids=pinyin_ids,
                            pinyin_lengths=pinyin_lengths
                        )
                        tmp_eval_loss = outputs[0]
                        tmp_eval_det_loss = outputs[1]
                        tmp_eval_cor_loss = outputs[2]
                        logits = outputs[-2]
                    
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_det_loss += tmp_eval_det_loss.mean().item()
                    eval_cor_loss += tmp_eval_cor_loss.mean().item()
                    
                    # 获取预测结果
                    _, prd_ids = torch.max(logits, -1)
                    prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
                    src_ids = src_ids.cpu().numpy().tolist()
                    trg_ids = trg_ids.cpu().numpy().tolist()
                    
                    # 解码为文本
                    for s, t, p in zip(src_ids, trg_ids, prd_ids):
                        mapped_src = []
                        mapped_trg = []
                        mapped_prd = []
                        for st, tt, pt in zip(s, t, p):
                            if st == tokenizer.sep_token_id or st == tokenizer.cls_token_id:
                                continue
                            mapped_trg.append(tt)
                            mapped_src.append(st)
                            mapped_prd.append(pt if st != pt else st)
                        
                        all_inputs.append(tokenizer.convert_ids_to_tokens(mapped_src, skip_special_tokens=True))
                        all_labels.append(tokenizer.convert_ids_to_tokens(mapped_trg, skip_special_tokens=True))
                        all_predictions.append(tokenizer.convert_ids_to_tokens(mapped_prd, skip_special_tokens=True))
                
                # 计算指标
                eval_loss = eval_loss / len(eval_dataloader)
                eval_det_loss = eval_det_loss / len(eval_dataloader)
                eval_cor_loss = eval_cor_loss / len(eval_dataloader)
                
                # 显示预测示例
                logger.info("\n预测示例:")
                for i in range(min(3, len(all_inputs))):
                    logger.info(f"输入: {''.join(all_inputs[i])}")
                    logger.info(f"标签: {''.join(all_labels[i])}")
                    logger.info(f"预测: {''.join(all_predictions[i])}\n")
                
                # 计算检测和纠正指标
                det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(
                    all_inputs, all_predictions, all_labels, logger
                )
                cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(
                    all_inputs, all_predictions, all_labels, logger
                )
                
                logger.info(f"检测 - Acc: {det_acc:.4f}, P: {det_precision:.4f}, R: {det_recall:.4f}, F1: {det_f1:.4f}")
                logger.info(f"纠正 - Acc: {cor_acc:.4f}, P: {cor_precision:.4f}, R: {cor_recall:.4f}, F1: {cor_f1:.4f}")
                
                # 记录到 TensorBoard
                tensorboardx_witer.add_scalar("loss/train", train_loss / (step + 1), global_step)
                tensorboardx_witer.add_scalar("loss/eval", eval_loss, global_step)
                tensorboardx_witer.add_scalar("det/f1", det_f1, global_step)
                tensorboardx_witer.add_scalar("cor/f1", cor_f1, global_step)
                tensorboardx_witer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                
                # 保存最佳模型
                if cor_f1 > best_f1:
                    best_f1 = cor_f1
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(
                        args.model_save_path,
                        f"best_model_step-{global_step}_f1-{cor_f1:.4f}.bin"
                    )
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info(f"✅ 保存最佳模型: F1={cor_f1:.4f}")
                    
                    # 保存配置
                    model_to_save.config.save_pretrained(save_directory=args.model_save_path)
                    tokenizer.save_pretrained(save_directory=args.model_save_path)
                    args.flag_train = False
                    save_json(vars(args), os.path.join(args.model_save_path, "csc.config"), mode="w")
    
    logger.info(f"\n训练完成！最佳 F1: {best_f1:.4f}")
    tensorboardx_witer.close()


if __name__ == "__main__":
    train_csc()
