# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
配置文件：基于 MacBERT 与拼音特征融合的中文拼写纠错模型
作者：[你的名字]
毕业设计：《基于 MacBERT 与拼音特征融合的中文拼写错误检测与纠正系统设计与实现》

【论文对应】第4章 实验与分析 - 4.1 实验设置
"""

from argparse import Namespace
import platform


# ========== Windows 配置（用于本地调试）==========
if platform.system().lower() == "windows":
    csc_config = {
        # ========== 模型配置 ==========
        "pretrained_model_name_or_path": "hfl/chinese-macbert-base",  # MacBERT 预训练模型
        
        # ========== 数据集配置 ==========
        # 【论文对应】第4章 - 4.1.1 数据集
        # 使用 SIGHAN 2015 数据集（中文拼写纠错的标准数据集）
        # ⭐ 使用绝对路径（避免相对路径在不同位置执行时出错）
        "path_train": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.train.json",
        "path_dev": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.dev.json",
        "path_tet": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.dev.json",
        "model_save_path": r"E:\GraduationDesign\macro-correct\macro_correct\output\text_correction",
        "task_name": "sighan2015_pinyin_fusion",  # ⭐ 任务名称（会作为模型保存目录）
        
        # ========== 训练超参数 ==========
        # 【论文对应】第4章 - 4.1.2 训练参数
        "do_lower_case": True,              # 是否转小写（中文不影响）
        "do_train": True,                   # 是否训练
        "do_eval": True,                    # 是否验证
        "do_test": True,                    # 是否测试
        
        "train_batch_size": 8,              # ⚠️ Windows 显存小，用小 batch
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,   # 梯度累积（相当于 batch=8*4=32）
        "learning_rate": 3e-5,              # 学习率（Adam 优化器）
        "max_train_steps": 100,            # ⚠️ 快速测试用 1000 步
        "num_train_epochs": None,           # 优先使用 max_train_steps
        "warmup_proportion": 0.1,           # 前 10% 步数用于 warmup
        "num_warmup_steps": None,
        
        "max_seq_length": 128,              # 最大序列长度
        "max_grad_norm": 1.0,               # 梯度裁剪
        "weight_decay": 5e-4,               # 权重衰减（正则化）
        "save_steps": 100,                  # 每 100 步验证一次
        "seed": 42,                         # 随机种子（保证可复现）
        
        # ========== 学习率调度器 ==========
        "lr_scheduler_type": "linear",      # 学习率衰减策略
        
        # ========== 损失函数配置 ==========
        # 【论文对应】第3章 - 3.3.3 损失函数
        "loss_type": "BCE",                 # 检测损失类型（二元交叉熵）
        "loss_det_rate": 0.3,               # ⭐ 检测损失权重（30% 检测 + 70% 纠正）
        
        # ========== 拼音特征配置（核心创新）==========
        # 【论文对应】第3章 - 3.2 拼音特征编码
        "use_pinyin": False,                 # ⭐⭐⭐ 消融实验开关：True=使用拼音, False=不使用拼音
        "fusion_type": "gate",              # ⭐ 融合方式："gate", "attention", "bilinear", "add"
        "pinyin_vocab_size": 500,           # 拼音字符词表大小
        "pinyin_embed_dim": 128,            # 拼音嵌入维度
        
        # ========== 其他配置 ==========
        "flag_fast_tokenizer": True,        # 使用快速 Tokenizer
        "flag_train": True,                 # 训练模式（会启用梯度检查点）
        "flag_fp16": False,                 # ⚠️ Windows 不建议用混合精度
        "flag_cuda": True,                  # 使用 GPU
        "flag_skip": True,                  # 跳过特殊字符
        "num_workers": 0,                   # DataLoader 线程数（Windows 用 0）
        "CUDA_VISIBLE_DEVICES": "0",        # 使用第 0 块 GPU
        "USE_TORCH": "1"
    }

# ========== Linux 配置（用于服务器训练）==========
else:
    csc_config = {
        # ========== 模型配置 ==========
        "pretrained_model_name_or_path": "hfl/chinese-macbert-base",
        
        # ========== 数据集配置 ==========
        # ⭐ Linux 系统需要修改为你的实际路径
        "path_train": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.train.json",
        "path_dev": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.dev.json",
        "path_tet": r"E:\GraduationDesign\macro-correct\macro_correct\corpus\text_correction\sighan\sighan2015.dev.json",
        "model_save_path": r"E:\GraduationDesign\macro-correct\macro_correct\output\text_correction",
        "task_name": "sighan2015_pinyin_fusion",
        
        # ========== 训练超参数 ==========
        "do_lower_case": True,
        "do_train": True,
        "do_eval": True,
        "do_test": True,
        
        "train_batch_size": 32,             # 服务器显存大，用大 batch
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 2,   # 相当于 batch=32*2=64
        "learning_rate": 3e-5,
        "max_train_steps": 100,            # 不限制步数None
        "num_train_epochs": 10,             # 训练 10 个 epoch
        "warmup_proportion": 0.1,
        "num_warmup_steps": None,
        
        "max_seq_length": 128,
        "max_grad_norm": 1.0,
        "weight_decay": 5e-4,
        "save_steps": 500,                  # 每 500 步验证一次
        "seed": 42,
        
        # ========== 学习率调度器 ==========
        "lr_scheduler_type": "cosine",      # 余弦衰减（效果更好）
        
        # ========== 损失函数配置 ==========
        "loss_type": "BCE",
        "loss_det_rate": 0.3,
        
        # ========== 拼音特征配置 ==========
        "use_pinyin": True,                 # ⭐⭐⭐ 消融实验开关
        "fusion_type": "gate",
        "pinyin_vocab_size": 500,
        "pinyin_embed_dim": 128,
        
        # ========== 其他配置 ==========
        "flag_fast_tokenizer": True,
        "flag_train": True,
        "flag_fp16": True,                  # 服务器可以用混合精度
        "flag_cuda": True,
        "flag_skip": True,
        "num_workers": 4,                   # 多线程加载数据
        "CUDA_VISIBLE_DEVICES": "0",
        "USE_TORCH": "1"
    }

# 转换为 Namespace 对象（方便用 args.xxx 访问）
csc_config = Namespace(**csc_config)


# ========== 配置说明（写进论文）==========
"""
【论文第4章 - 表4.1 实验参数设置】

| 参数名称 | 参数值 | 说明 |
|---------|--------|------|
| 预训练模型 | hfl/chinese-macbert-base | MacBERT 中文预训练模型 |
| 数据集 | SIGHAN 2015 | 中文拼写纠错标准数据集 |
| Batch Size | 32 (Linux) / 8 (Windows) | 批次大小 |
| 学习率 | 3e-5 | Adam 优化器学习率 |
| 训练轮数 | 10 epochs | 完整遍历数据集 10 次 |
| 最大序列长度 | 128 | 输入文本最大长度 |
| 损失权重 | 0.3 (检测) + 0.7 (纠正) | 检测和纠正损失的权重 |
| 融合方式 | gate | 门控融合机制 |
| 拼音嵌入维度 | 128 | 拼音特征向量维度 |

【核心创新参数】
- fusion_type: 特征融合方式（gate/attention/bilinear）
- loss_det_rate: 检测损失权重（平衡检测和纠正）
- pinyin_embed_dim: 拼音特征维度（影响模型表达能力）
"""
