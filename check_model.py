#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查本地是否已下载 MacBERT 模型
"""

import os
from pathlib import Path

print("=" * 80)
print("检查本地 MacBERT 模型")
print("=" * 80)

# 方法1：检查 HuggingFace 缓存
try:
    from transformers import file_utils
    cache_dir = file_utils.default_cache_path
    print(f"\n✅ HuggingFace 缓存目录: {cache_dir}")
    
    if os.path.exists(cache_dir):
        # 列出所有模型
        models = []
        for item in os.listdir(cache_dir):
            if 'macbert' in item.lower():
                models.append(item)
        
        if models:
            print(f"\n✅ 找到 MacBERT 相关模型:")
            for model in models:
                print(f"   - {model}")
        else:
            print("\n❌ 未找到 MacBERT 模型")
    else:
        print(f"❌ 缓存目录不存在: {cache_dir}")
except Exception as e:
    print(f"⚠️ 检查缓存失败: {e}")

# 方法2：尝试加载模型（不下载）
print("\n" + "-" * 80)
print("尝试加载 MacBERT 模型（仅检查本地）")
print("-" * 80)

try:
    from transformers import AutoConfig
    
    # 设置为仅本地模式（不从网络下载）
    config = AutoConfig.from_pretrained(
        "hfl/chinese-macbert-base",
        local_files_only=True  # 只使用本地文件
    )
    print("\n✅ 成功！本地已有 MacBERT 模型")
    print(f"   模型配置: {config}")
    
except Exception as e:
    print(f"\n❌ 本地没有 MacBERT 模型")
    print(f"   错误信息: {e}")

# 方法3：检查常见的本地模型路径
print("\n" + "-" * 80)
print("检查常见的本地模型路径")
print("-" * 80)

common_paths = [
    "./models/chinese-macbert-base",
    "./pretrained_models/chinese-macbert-base",
    "../models/chinese-macbert-base",
    "../../models/chinese-macbert-base",
]

found_local = False
for path in common_paths:
    if os.path.exists(path):
        print(f"✅ 找到本地模型: {path}")
        found_local = True
        break

if not found_local:
    print("❌ 未在常见路径找到本地模型")

# 总结和建议
print("\n" + "=" * 80)
print("总结和建议")
print("=" * 80)

print("""
如果本地没有模型，你有以下几种选择：

【方案1】使用国内镜像站下载（推荐）
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModel
model = AutoModel.from_pretrained("hfl/chinese-macbert-base")
```

【方案2】手动下载到本地
1. 访问: https://hf-mirror.com/hfl/chinese-macbert-base
2. 下载所有文件到本地目录（如 ./models/chinese-macbert-base）
3. 修改代码中的路径为本地路径

【方案3】跳过完整模型测试
- 只测试核心组件（PinyinEncoder、FusionLayer）
- 这些测试不需要下载模型，已经足够验证代码逻辑

【方案4】使用代理
如果你有代理，可以配置环境变量：
```bash
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
```
""")
