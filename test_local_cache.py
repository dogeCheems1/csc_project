#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证能否从本地缓存加载模型
"""

import os

# ⭐ 清除代理设置
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=" * 80)
print("测试：从本地缓存加载 MacBERT 模型")
print("=" * 80)

try:
    from transformers import AutoTokenizer, BertForMaskedLM
    
    print("\n1️⃣ 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "hfl/chinese-macbert-base",
        local_files_only=True  # 只使用本地缓存
    )
    print("   ✅ Tokenizer 加载成功")
    print(f"   词表大小: {tokenizer.vocab_size}")
    
    print("\n2️⃣ 加载模型...")
    model = BertForMaskedLM.from_pretrained(
        "hfl/chinese-macbert-base",
        local_files_only=True  # 只使用本地缓存
    )
    print("   ✅ 模型加载成功")
    print(f"   隐藏层维度: {model.config.hidden_size}")
    
    print("\n3️⃣ 测试推理...")
    test_text = "我爱中国"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    print(f"   ✅ 推理成功！输出形状: {outputs.logits.shape}")
    
    print("\n" + "=" * 80)
    print("🎉 测试通过！可以从本地缓存加载模型")
    print("=" * 80)
    print("\n现在可以运行训练脚本了：")
    print("python macro_correct/pytorch_user_models/csc/macbert4csc_pinyin/run_ablation_study.py")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ 测试失败")
    print("=" * 80)
    print(f"错误信息: {e}")
    print("\n可能的原因：")
    print("1. 本地缓存中没有模型（需要先下载）")
    print("2. 代理设置仍然有问题")
    print("\n解决方案：")
    print("1. 运行下载脚本：python download_macbert.py")
    print("2. 或者手动关闭系统代理后再试")
