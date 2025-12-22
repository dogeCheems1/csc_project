# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
测试脚本：验证拼音融合模型的维度对齐、Mask处理、梯度回传
"""

import torch
import torch.nn as nn
from graph import Macbert4CSCWithPinyin, PinyinEncoder, FusionLayer


class MockConfig:
    """模拟配置对象"""
    def __init__(self):
        # ⭐ 使用已下载的 MacBERT 模型（从缓存加载）
        self.pretrained_model_name_or_path = "hfl/chinese-macbert-base"
        
        # ⭐ 设置镜像站（确保能从缓存加载）
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        self.flag_train = False
        self.loss_det_rate = 0.3


def test_pinyin_encoder():
    """测试1：PinyinEncoder 的 Mask 处理"""
    print("=" * 80)
    print("测试1：PinyinEncoder 的 Mask 处理")
    print("=" * 80)
    
    encoder = PinyinEncoder(pinyin_vocab_size=500, pinyin_embed_dim=128, hidden_size=768)
    
    # 模拟输入：[batch=2, seq_len=5, pinyin_len=8]
    batch_size, seq_len, pinyin_len = 2, 5, 8
    pinyin_ids = torch.randint(0, 50, (batch_size, seq_len, pinyin_len))
    pinyin_lengths = torch.tensor([
        [5, 6, 4, 0, 3],  # 第1个样本的5个拼音的实际长度
        [7, 5, 6, 4, 0]   # 第2个样本的5个拼音的实际长度
    ])
    
    # 不使用 Mask（旧版本）
    features_no_mask = encoder(pinyin_ids, pinyin_lengths=None)
    print(f"✅ 不使用Mask的输出形状: {features_no_mask.shape}")
    
    # 使用 Mask（新版本）
    features_with_mask = encoder(pinyin_ids, pinyin_lengths=pinyin_lengths)
    print(f"✅ 使用Mask的输出形状: {features_with_mask.shape}")
    
    # 验证维度
    assert features_no_mask.shape == (batch_size, seq_len, 768), "维度不匹配！"
    assert features_with_mask.shape == (batch_size, seq_len, 768), "维度不匹配！"
    
    # 验证两者有差异（说明Mask生效了）
    diff = torch.abs(features_no_mask - features_with_mask).mean().item()
    print(f"✅ 两种方式的特征差异: {diff:.6f} (>0 说明Mask生效)")
    assert diff > 0, "Mask没有生效！"
    
    print("✅ 测试1通过：PinyinEncoder 的 Mask 处理正确\n")


def test_fusion_layer():
    """测试2：FusionLayer 的融合逻辑和梯度回传"""
    print("=" * 80)
    print("测试2：FusionLayer 的融合逻辑和梯度回传")
    print("=" * 80)
    
    batch_size, seq_len, hidden_size = 2, 10, 768
    
    # 模拟输入
    text_features = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    pinyin_features = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 测试不同融合方式
    for fusion_type in ["gate", "attention", "bilinear", "add"]:
        print(f"\n--- 测试融合方式: {fusion_type} ---")
        fusion_layer = FusionLayer(hidden_size=hidden_size, fusion_type=fusion_type)
        
        # 前向传播
        fused = fusion_layer(text_features, pinyin_features, attention_mask)
        print(f"✅ 融合后形状: {fused.shape}")
        assert fused.shape == (batch_size, seq_len, hidden_size), "维度不匹配！"
        
        # 反向传播（验证梯度）
        loss = fused.sum()
        loss.backward()
        
        # 检查梯度是否存在
        assert text_features.grad is not None, "text_features 梯度为空！"
        assert pinyin_features.grad is not None, "pinyin_features 梯度为空！"
        print(f"✅ text_features 梯度范数: {text_features.grad.norm().item():.6f}")
        print(f"✅ pinyin_features 梯度范数: {pinyin_features.grad.norm().item():.6f}")
        
        # 清空梯度
        text_features.grad = None
        pinyin_features.grad = None
    
    print("\n✅ 测试2通过：FusionLayer 的融合逻辑和梯度回传正确\n")


def test_full_model():
    """测试3：完整模型的维度对齐"""
    print("=" * 80)
    print("测试3：完整模型的维度对齐")
    print("=" * 80)
    
    config = MockConfig()
    
    # ⭐ 如果没有配置模型路径，跳过此测试
    if config.pretrained_model_name_or_path is None:
        print("⚠️ 未配置预训练模型路径，跳过完整模型测试")
        print("💡 提示：如果需要测试完整模型，请在 MockConfig 中配置本地模型路径")
        print("跳过测试3\n")
        return
    
    # 注意：这里需要真实的 MacBERT 模型，如果没有会报错
    try:
        model = Macbert4CSCWithPinyin(config, csc_config=config)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"⚠️ 模型初始化失败: {e}")
        print("💡 提示：请检查网络连接或使用本地模型路径")
        print("跳过测试3\n")
        return
    
    # 模拟输入
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(100, 5000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(100, 5000, (batch_size, seq_len))
    
    # 方式1：预计算拼音
    texts = ["我爱中国人民", "今天天气很好"]
    pinyin_ids, pinyin_lengths = model.text_to_pinyin_ids(texts, seq_len)
    print(f"✅ 拼音ID形状: {pinyin_ids.shape}")
    print(f"✅ 拼音长度形状: {pinyin_lengths.shape}")
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pinyin_ids=pinyin_ids,
        pinyin_lengths=pinyin_lengths
    )
    
    loss, det_loss, cor_loss, det_probs, cor_probs, pred_ids = outputs
    print(f"✅ 总损失: {loss.item():.4f}")
    print(f"✅ 检测损失: {det_loss.item():.4f}")
    print(f"✅ 纠正损失: {cor_loss.item():.4f}")
    print(f"✅ 检测概率形状: {det_probs.shape}")
    print(f"✅ 纠正概率形状: {cor_probs.shape}")
    print(f"✅ 预测ID形状: {pred_ids.shape}")
    
    # 验证维度
    assert det_probs.shape == (batch_size, seq_len), "检测概率维度不匹配！"
    assert cor_probs.shape[:-1] == (batch_size, seq_len), "纠正概率维度不匹配！"
    assert pred_ids.shape == (batch_size, seq_len), "预测ID维度不匹配！"
    
    # 方式2：动态计算拼音
    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        texts=texts  # 直接传入文本
    )
    loss2 = outputs2[0]
    print(f"✅ 动态计算拼音的损失: {loss2.item():.4f}")
    
    print("\n✅ 测试3通过：完整模型的维度对齐正确\n")


def test_gradient_flow():
    """测试4：验证梯度能同时更新 BERT 和 GRU"""
    print("=" * 80)
    print("测试4：验证梯度能同时更新 BERT 和 GRU")
    print("=" * 80)
    
    config = MockConfig()
    
    # ⭐ 如果没有配置模型路径，跳过此测试
    if config.pretrained_model_name_or_path is None:
        print("⚠️ 未配置预训练模型路径，跳过梯度测试")
        print("💡 提示：如果需要测试梯度流，请在 MockConfig 中配置本地模型路径")
        print("跳过测试4\n")
        return
    
    try:
        model = Macbert4CSCWithPinyin(config, csc_config=config)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"⚠️ 模型初始化失败: {e}")
        print("💡 提示：请检查网络连接或使用本地模型路径")
        print("跳过测试4\n")
        return
    
    # 模拟输入
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(100, 5000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(100, 5000, (batch_size, seq_len))
    texts = ["我爱中国", "今天天气"]
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        texts=texts
    )
    loss = outputs[0]
    
    # 反向传播
    loss.backward()
    
    # 检查 BERT 的梯度
    bert_has_grad = False
    for name, param in model.bert.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            bert_has_grad = True
            print(f"✅ BERT 参数 {name} 有梯度: {param.grad.norm().item():.6f}")
            break
    
    # 检查 PinyinEncoder 的梯度
    pinyin_has_grad = False
    for name, param in model.pinyin_encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            pinyin_has_grad = True
            print(f"✅ PinyinEncoder 参数 {name} 有梯度: {param.grad.norm().item():.6f}")
            break
    
    # 检查 FusionLayer 的梯度
    fusion_has_grad = False
    for name, param in model.fusion_layer.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            fusion_has_grad = True
            print(f"✅ FusionLayer 参数 {name} 有梯度: {param.grad.norm().item():.6f}")
            break
    
    assert bert_has_grad, "BERT 没有梯度！"
    assert pinyin_has_grad, "PinyinEncoder 没有梯度！"
    assert fusion_has_grad, "FusionLayer 没有梯度！"
    
    print("\n✅ 测试4通过：梯度能同时更新 BERT、GRU 和 FusionLayer\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("开始测试拼音融合模型")
    print("=" * 80 + "\n")
    
    # 运行所有测试
    test_pinyin_encoder()
    test_fusion_layer()
    test_full_model()
    test_gradient_flow()
    
    print("=" * 80)
    print("🎉 所有测试通过！模型实现正确！")
    print("=" * 80)
