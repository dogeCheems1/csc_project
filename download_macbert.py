#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用国内镜像站下载 MacBERT 模型
"""

import os

print("=" * 80)
print("开始下载 MacBERT 模型（使用国内镜像站）")
print("=" * 80)

# ⭐ 关键：设置国内镜像站（hf-mirror.com）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("\n✅ 已设置镜像站: https://hf-mirror.com")
print("📥 开始下载模型文件...\n")

try:
    from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
    
    model_name = "hfl/chinese-macbert-base"
    
    # 下载 Tokenizer
    print("1️⃣ 下载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✅ Tokenizer 下载完成")
    
    # 下载模型
    print("\n2️⃣ 下载模型权重（这可能需要几分钟）...")
    model = BertForMaskedLM.from_pretrained(model_name)
    print("   ✅ 模型下载完成")
    
    # 验证模型
    print("\n3️⃣ 验证模型...")
    print(f"   - 模型配置: {model.config}")
    print(f"   - 词表大小: {tokenizer.vocab_size}")
    print(f"   - 隐藏层维度: {model.config.hidden_size}")
    
    # 测试模型
    print("\n4️⃣ 测试模型推理...")
    test_text = "我爱中国"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    print(f"   ✅ 模型推理成功！输出形状: {outputs.logits.shape}")
    
    print("\n" + "=" * 80)
    print("🎉 模型下载并验证成功！")
    print("=" * 80)
    print(f"\n模型已缓存到: {os.path.expanduser('~/.cache/huggingface/hub')}")
    print("现在可以运行测试脚本了：python test_model.py")
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ 下载失败")
    print("=" * 80)
    print(f"错误信息: {e}")
    print("\n可能的原因：")
    print("1. 网络连接问题（即使是镜像站也需要网络）")
    print("2. 代理配置冲突（如果有代理，可能需要关闭）")
    print("3. 磁盘空间不足（模型约 400MB）")
    print("\n建议：")
    print("- 检查网络连接")
    print("- 尝试关闭代理：set HTTP_PROXY= && set HTTPS_PROXY=")
    print("- 或者手动下载：https://hf-mirror.com/hfl/chinese-macbert-base/tree/main")
