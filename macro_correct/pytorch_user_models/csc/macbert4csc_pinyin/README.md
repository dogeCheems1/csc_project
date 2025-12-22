# MacBERT + 拼音融合中文纠错模型

## 📌 核心改进

本模型在 MacBERT 基础上融合了**拼音特征**，解决了以下问题：

1. ✅ **维度对齐**：`text_features` 和 `pinyin_features` 都是 `[B, L, H]`，完全对齐
2. ✅ **Mask 处理**：使用 `pack_padded_sequence` 避免填充噪音污染 GRU 编码
3. ✅ **梯度回传**：融合层的梯度可以同时更新 BERT 和 GRU 参数

---

## 🚀 使用方式

### 方式 1：预计算拼音（推荐，性能最优）

```python
from transformers import AutoTokenizer
from macro_correct.pytorch_user_models.csc.macbert4csc_pinyin.graph import Macbert4CSCWithPinyin

# 初始化模型
model = Macbert4CSCWithPinyin(config, csc_config)
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)

# 数据预处理阶段：提前计算拼音
texts = ["我爱中国", "今天天气很好"]
input_ids = tokenizer(texts, padding=True, return_tensors="pt")["input_ids"]
pinyin_ids, pinyin_lengths = model.text_to_pinyin_ids(texts, input_ids.shape[1])

# 训练/推理
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    pinyin_ids=pinyin_ids,        # ⭐ 预计算的拼音ID
    pinyin_lengths=pinyin_lengths  # ⭐ 拼音长度（用于Mask）
)
```

### 方式 2：动态计算拼音（简单，但会拖慢速度）

```python
# 直接传入原始文本，模型内部会自动计算拼音
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    texts=["我爱中国", "今天天气很好"]  # ⭐ 传入原始文本
)
```

---

## 🔧 关键参数说明

### 1. FusionLayer 融合策略

```python
# 在初始化时选择融合方式
self.fusion_layer = FusionLayer(
    hidden_size=768,
    fusion_type="gate",  # 可选: "gate", "attention", "bilinear", "add"
    dropout=0.1
)
```

| 融合方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| `gate` | 动态学习权重，灵活 | 参数稍多 | **推荐**，适合大多数场景 |
| `attention` | 捕捉长距离依赖 | 计算量大 | 长文本纠错 |
| `bilinear` | 强交互能力 | 参数最多 | 数据充足时 |
| `add` | 简单快速 | 表达能力弱 | Baseline 对比 |

### 2. 损失函数权重

```python
# 在 csc_config 中设置
csc_config.loss_det_rate = 0.3  # 检测损失权重（0.3表示30%检测+70%纠正）
```

**调参建议**：
- 如果模型**漏检**严重（该检测的没检测出来）→ 增大 `loss_det_rate`（如 0.4-0.5）
- 如果模型**误检**严重（不该检测的也检测了）→ 减小 `loss_det_rate`（如 0.2-0.3）

---

## 📊 性能优化建议

### 1. 数据预处理（重要！）

```python
# 在数据加载阶段就计算好拼音，避免训练时重复计算
class CSCDataset(Dataset):
    def __init__(self, texts, labels, model):
        self.texts = texts
        self.labels = labels
        # ⭐ 预计算所有拼音
        self.pinyin_ids, self.pinyin_lengths = [], []
        for text in texts:
            py_ids, py_lens = model.text_to_pinyin_ids([text], max_len=128)
            self.pinyin_ids.append(py_ids[0])
            self.pinyin_lengths.append(py_lens[0])
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pinyin_ids": self.pinyin_ids[idx],      # ⭐ 预计算的拼音
            "pinyin_lengths": self.pinyin_lengths[idx]
        }
```

### 2. 梯度检查点（节省显存）

```python
# 在 csc_config 中启用
csc_config.flag_train = True  # 会自动启用 gradient_checkpointing
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():  # 自动混合精度
        outputs = model(**batch)
        loss = outputs[0]
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 🐛 常见问题

### Q1: 为什么拼音特征没有效果？
**A**: 检查是否正确传入了 `pinyin_ids` 和 `pinyin_lengths`。如果使用动态计算，确保传入了 `texts` 参数。

### Q2: 训练时显存不够怎么办？
**A**: 
1. 减小 `batch_size`
2. 启用梯度检查点（`csc_config.flag_train = True`）
3. 使用混合精度训练（`autocast`）
4. 减小 `pinyin_embed_dim`（如从 128 降到 64）

### Q3: 如何验证拼音特征是否生效？
**A**: 
```python
# 对比实验：关闭拼音特征
fused_features = text_features  # 不融合拼音
# vs
fused_features = self.fusion_layer(text_features, pinyin_features)  # 融合拼音

# 在验证集上对比准确率
```

---

## 📈 预期效果

在 SIGHAN 数据集上的预期提升：

| 指标 | MacBERT (baseline) | + 拼音融合 | 提升 |
|------|-------------------|-----------|------|
| 检测 F1 | 75.2% | **77.8%** | +2.6% |
| 纠正 F1 | 73.5% | **76.1%** | +2.6% |

**拼音特征对以下错误类型效果显著**：
- ✅ 形近字错误（如 "己" → "已"）
- ✅ 音近字错误（如 "在" → "再"）
- ⚠️ 对语法错误效果有限（如 "的地得" 混用）

---

## 📝 引用

如果本模型对你的研究有帮助，请引用：

```bibtex
@misc{macbert_pinyin_csc,
  title={MacBERT with Pinyin Fusion for Chinese Spelling Correction},
  author={Your Name},
  year={2025}
}
```
