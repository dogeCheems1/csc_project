#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验脚本：对比是否使用拼音特征

【论文对应】第4章 实验与分析 - 4.4 消融实验

实验设计：
1. Baseline：只用 MacBERT（use_pinyin=False）
2. 改进版：MacBERT + 拼音融合（use_pinyin=True）

目的：证明拼音特征确实能提升模型性能
"""

import os
import sys
import json
from copy import deepcopy

# ⭐ 关键：清除代理设置（避免与本地代理冲突）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 设置 HuggingFace 为离线模式（优先使用本地缓存）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加项目根目录到路径
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.config import csc_config
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.train import train_csc


def run_ablation_study():
    """
    运行消融实验：对比使用和不使用拼音特征
    """
    print("=" * 80)
    print("消融实验：验证拼音特征的有效性")
    print("=" * 80)
    
    # 保存原始配置
    original_config = deepcopy(vars(csc_config))
    
    # ========== 实验 1：Baseline（不使用拼音）==========
    print("\n" + "=" * 80)
    print("实验 1：Baseline（只用 MacBERT，不使用拼音特征）")
    print("=" * 80)
    
    # 修改配置
    csc_config.use_pinyin = False  # ⭐ 关闭拼音特征
    csc_config.task_name = "sighan2015_baseline_no_pinyin"  # 修改任务名称
    csc_config.max_train_steps = 500  # 快速测试用 500 步
    
    print(f"配置: use_pinyin={csc_config.use_pinyin}")
    print(f"任务名称: {csc_config.task_name}")
    print(f"训练步数: {csc_config.max_train_steps}")
    
    # 开始训练
    print("\n开始训练 Baseline 模型...")
    train_csc()
    
    # ========== 实验 2：改进版（使用拼音）==========
    print("\n" + "=" * 80)
    print("实验 2：改进版（MacBERT + 拼音融合）")
    print("=" * 80)
    
    # 恢复配置并修改
    for key, value in original_config.items():
        setattr(csc_config, key, value)
    
    csc_config.use_pinyin = True  # ⭐ 开启拼音特征
    csc_config.task_name = "sighan2015_with_pinyin"  # 修改任务名称
    csc_config.max_train_steps = 500  # 快速测试用 500 步
    
    print(f"配置: use_pinyin={csc_config.use_pinyin}")
    print(f"任务名称: {csc_config.task_name}")
    print(f"训练步数: {csc_config.max_train_steps}")
    
    # 开始训练
    print("\n开始训练改进版模型...")
    train_csc()
    
    # ========== 对比结果 ==========
    print("\n" + "=" * 80)
    print("实验完成！现在对比两个模型的结果")
    print("=" * 80)
    
    # 读取两个模型的训练日志
    baseline_log = "../../../output/text_correction/sighan2015_baseline_no_pinyin/train.log"
    improved_log = "../../../output/text_correction/sighan2015_with_pinyin/train.log"
    
    print("\n📊 结果对比：")
    print("\n1. Baseline（不使用拼音）:")
    print(f"   日志文件: {baseline_log}")
    print("   请查看最后的 F1 分数")
    
    print("\n2. 改进版（使用拼音）:")
    print(f"   日志文件: {improved_log}")
    print("   请查看最后的 F1 分数")
    
    print("\n💡 如何对比：")
    print("1. 打开两个日志文件")
    print("2. 搜索 'Sentence Level correction' 找到最终的 F1 分数")
    print("3. 对比两个模型的 F1，如果改进版更高，说明拼音特征有效！")
    
    print("\n📝 论文写作建议：")
    print("在第4章 - 4.4 消融实验中，创建一个表格：")
    print("""
    | 模型 | 检测 F1 | 纠正 F1 | 提升 |
    |------|---------|---------|------|
    | Baseline（不使用拼音） | 0.XX | 0.XX | - |
    | 改进版（使用拼音） | 0.XX | 0.XX | +X.X% |
    """)


if __name__ == "__main__":
    run_ablation_study()
