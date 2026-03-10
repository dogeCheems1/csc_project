#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速诊断脚本：验证拼音特征是否真正参与训练
"""

import torch
import sys
import os

# 添加项目路径
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.config import csc_config
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.graph import Macbert4CSCWithPinyin

def check_pinyin_usage():
    """检查拼音特征是否真正被使用"""
    print("=" * 80)
    print("诊断：拼音特征是否真正参与训练")
    print("=" * 80)
    
    # 1. 检查配置
    print(f"\n1. 配置检查:")
    print(f"   use_pinyin = {csc_config.use_pinyin}")
    print(f"   fusion_type = {csc_config.fusion_type}")
    
    # 2. 初始化模型
    print(f"\n2. 初始化模型...")
    model = Macbert4CSCWithPinyin(config=csc_config, csc_config=csc_config)
    print(f"   模型内部 use_pinyin = {model.use_pinyin}")
    
    # 3. 创建模拟输入
    batch_size, seq_len = 2, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_ids = torch.randint(100, 5000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    labels = torch.randint(100, 5000, (batch_size, seq_len)).to(device)
    texts = ["我爱中国", "今天天气很好"]
    
    # 4. 前向传播（开启拼音）
    print(f"\n3. 前向传播（use_pinyin={model.use_pinyin})...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            texts=texts
        )
    
    # 5. 检查拼音编码器参数是否有梯度（需要训练模式）
    print(f"\n4. 检查拼音编码器参数:")
    pinyin_params = list(model.pinyin_encoder.parameters())
    fusion_params = list(model.fusion_layer.parameters())
    
    print(f"   拼音编码器参数数量: {len(pinyin_params)}")
    print(f"   融合层参数数量: {len(fusion_params)}")
    
    # 6. 模拟一次反向传播，看梯度
    print(f"\n5. 模拟反向传播，检查梯度...")
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        texts=texts
    )
    loss = outputs[0]
    loss.backward()
    
    # 检查梯度
    pinyin_has_grad = False
    for i, param in enumerate(pinyin_params):
        if param.grad is not None and param.grad.abs().sum() > 1e-6:
            pinyin_has_grad = True
            print(f"   [OK] 拼音编码器参数 {i} 有梯度: {param.grad.norm().item():.6f}")
            break
    
    fusion_has_grad = False
    for i, param in enumerate(fusion_params):
        if param.grad is not None and param.grad.abs().sum() > 1e-6:
            fusion_has_grad = True
            print(f"   [OK] 融合层参数 {i} 有梯度: {param.grad.norm().item():.6f}")
            break
    
    if not pinyin_has_grad:
        print(f"   [WARNING] 拼音编码器没有梯度！可能没有被训练。")
    if not fusion_has_grad:
        print(f"   [WARNING] 融合层没有梯度！可能没有被训练。")
    
    # 7. 检查门控权重（如果是gate融合）
    if csc_config.fusion_type == "gate":
        print(f"\n6. 检查门控权重分布:")
        # 重新前向传播，但这次hook住gate的输出
        gate_values = []
        def hook_gate(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                gate_values.append(output.mean().item())
        
        # 注册hook
        handle = model.fusion_layer.gate[-1].register_forward_hook(hook_gate)
        
        model.eval()
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                texts=texts
            )
        
        handle.remove()
        
        if gate_values:
            avg_gate = sum(gate_values) / len(gate_values)
            print(f"   平均门控权重: {avg_gate:.4f}")
            if avg_gate > 0.9:
                print(f"   [WARNING] 门控权重接近1，拼音特征可能被忽略！")
            elif avg_gate < 0.1:
                print(f"   [WARNING] 门控权重接近0，文本特征可能被忽略！")
            else:
                print(f"   [OK] 门控权重在合理范围，两种特征都在使用。")
    
    print(f"\n" + "=" * 80)
    print("诊断完成！")
    print("=" * 80)

if __name__ == "__main__":
    check_pinyin_usage()

