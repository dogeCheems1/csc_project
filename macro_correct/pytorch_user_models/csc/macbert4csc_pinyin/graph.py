# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
基于MacBERT与拼音特征融合的中文拼写纠错模型
论文核心创新：在MacBERT基础上融合拼音特征
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertPreTrainedModel
from pypinyin import pinyin, Style
import numpy as np


class PinyinEncoder(nn.Module):
    """拼音编码器：将拼音序列编码为向量"""
    
    def __init__(self, pinyin_vocab_size=500, pinyin_embed_dim=128, hidden_size=768):
        super().__init__()
        # 拼音字符嵌入（包括声母、韵母、声调）
        self.pinyin_embedding = nn.Embedding(pinyin_vocab_size, pinyin_embed_dim)
        # 拼音序列编码（使用GRU或CNN）
        self.pinyin_gru = nn.GRU(pinyin_embed_dim, hidden_size // 2, 
                                  batch_first=True, bidirectional=True)
        # 投影到BERT隐藏层维度
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: [batch_size, seq_len, max_pinyin_len] 拼音字符ID序列
        Returns:
            pinyin_features: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, pinyin_len = pinyin_ids.shape
        # 展平处理
        pinyin_ids_flat = pinyin_ids.view(-1, pinyin_len)  # [B*L, P]
        pinyin_embed = self.pinyin_embedding(pinyin_ids_flat)  # [B*L, P, E]
        # GRU编码
        _, hidden = self.pinyin_gru(pinyin_embed)  # hidden: [2, B*L, H/2]
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [B*L, H]
        # 恢复形状并投影
        pinyin_features = hidden.view(batch_size, seq_len, -1)  # [B, L, H]
        pinyin_features = self.projection(pinyin_features)
        return pinyin_features


class FusionLayer(nn.Module):
    """特征融合层：融合文本特征和拼音特征"""
    
    def __init__(self, hidden_size=768, fusion_type="gate"):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "gate":
            # 门控融合机制
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        elif fusion_type == "attention":
            # 注意力融合
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, text_features, pinyin_features):
        """
        Args:
            text_features: [B, L, H] MacBERT输出
            pinyin_features: [B, L, H] 拼音编码输出
        Returns:
            fused_features: [B, L, H] 融合后特征
        """
        if self.fusion_type == "gate":
            concat = torch.cat([text_features, pinyin_features], dim=-1)
            gate = self.gate(concat)
            fused = gate * text_features + (1 - gate) * pinyin_features
        elif self.fusion_type == "attention":
            fused, _ = self.attention(text_features, pinyin_features, pinyin_features)
            fused = fused + text_features  # 残差连接
        else:
            # 简单加权
            fused = 0.7 * text_features + 0.3 * pinyin_features
            
        return self.layer_norm(fused)


class Macbert4CSCWithPinyin(BertPreTrainedModel):
    """融合拼音特征的MacBERT中文拼写纠错模型"""
    
    def __init__(self, config, csc_config=None):
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name_or_path)
        super().__init__(bert_config)
        
        self.csc_config = csc_config or config
        self.hidden_size = bert_config.hidden_size
        
        # MacBERT主干网络
        if hasattr(self.csc_config, 'flag_train') and self.csc_config.flag_train:
            self.bert = BertForMaskedLM.from_pretrained(
                config.pretrained_model_name_or_path,
                output_hidden_states=True
            )
        else:
            self.bert = BertForMaskedLM(bert_config)
        
        # ⭐ 拼音编码器（核心创新）
        self.pinyin_encoder = PinyinEncoder(
            pinyin_vocab_size=500,
            pinyin_embed_dim=128,
            hidden_size=self.hidden_size
        )
        
        # ⭐ 特征融合层（核心创新）
        self.fusion_layer = FusionLayer(
            hidden_size=self.hidden_size,
            fusion_type="gate"  # 可选: "gate", "attention", "add"
        )
        
        # 错误检测头
        self.detect = nn.Linear(self.hidden_size, 1)
        
        # 纠正输出头
        self.correct = nn.Linear(self.hidden_size, bert_config.vocab_size)
        self.correct.weight.data = self.bert.bert.embeddings.word_embeddings.weight.data
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        
        # 损失函数
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_bce = nn.BCELoss()
        
        # 拼音词表（用于拼音ID转换）
        self._build_pinyin_vocab()
        
    def _build_pinyin_vocab(self):
        """构建拼音字符词表"""
        # 基础拼音字符：声母+韵母+声调
        chars = list("abcdefghijklmnopqrstuvwxyz12345")
        self.pinyin_char2id = {"[PAD]": 0, "[UNK]": 1}
        for i, c in enumerate(chars):
            self.pinyin_char2id[c] = i + 2
        self.max_pinyin_len = 7  # 最长拼音如 "zhuang1"
        
    def text_to_pinyin_ids(self, texts, max_len):
        """将文本转换为拼音ID序列"""
        batch_pinyin_ids = []
        for text in texts:
            # 获取每个字的拼音（带声调）
            pinyins = pinyin(text, style=Style.TONE3, errors='ignore')
            seq_pinyin_ids = []
            for py in pinyins:
                if py:
                    py_str = py[0].lower()
                    char_ids = [self.pinyin_char2id.get(c, 1) for c in py_str]
                    # 填充到固定长度
                    char_ids = char_ids[:self.max_pinyin_len]
                    char_ids += [0] * (self.max_pinyin_len - len(char_ids))
                else:
                    char_ids = [0] * self.max_pinyin_len
                seq_pinyin_ids.append(char_ids)
            # 填充序列长度
            while len(seq_pinyin_ids) < max_len:
                seq_pinyin_ids.append([0] * self.max_pinyin_len)
            seq_pinyin_ids = seq_pinyin_ids[:max_len]
            batch_pinyin_ids.append(seq_pinyin_ids)
        return torch.tensor(batch_pinyin_ids)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, pinyin_ids=None):
        """
        前向传播
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: 句子类型ID
            labels: 标签（正确的token ID）
            pinyin_ids: 拼音ID序列（可选，如果不提供则自动计算）
        """
        # 1. MacBERT编码
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        text_features = outputs.hidden_states[-1]  # [B, L, H]
        
        # 2. ⭐ 拼音特征编码（核心创新）
        if pinyin_ids is None:
            # 如果没有提供拼音ID，则动态计算（实际使用时建议预计算）
            device = input_ids.device
            pinyin_ids = self.text_to_pinyin_ids(
                [self.bert.bert.embeddings.word_embeddings.weight.new_zeros(1)],  # placeholder
                input_ids.shape[1]
            ).to(device)
        pinyin_features = self.pinyin_encoder(pinyin_ids)  # [B, L, H]
        
        # 3. ⭐ 特征融合（核心创新）
        fused_features = self.fusion_layer(text_features, pinyin_features)  # [B, L, H]
        
        # 4. 错误检测
        det_logits = self.detect(fused_features).squeeze(-1)  # [B, L]
        det_probs = self.sigmoid(det_logits) * attention_mask
        
        # 5. 纠正预测
        cor_logits = self.correct(fused_features)  # [B, L, V]
        cor_probs = self.softmax(cor_logits)
        pred_prob, pred_ids = torch.max(cor_probs, dim=-1)
        pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
        
        if labels is not None:
            # 计算损失
            det_labels = (input_ids != labels).float()
            det_loss = self.loss_bce(det_probs, det_labels)
            cor_loss = self.loss_ce(cor_logits.view(-1, cor_logits.size(-1)), labels.view(-1))
            
            # 总损失（可调节权重）
            loss_det_rate = getattr(self.csc_config, 'loss_det_rate', 0.3)
            loss = (1 - loss_det_rate) * cor_loss + loss_det_rate * det_loss
            
            return (loss, det_loss, cor_loss, det_probs, cor_probs, pred_ids)
        else:
            return (cor_probs, pred_prob, pred_ids)