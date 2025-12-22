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
    """拼音编码器：将拼音序列编码为向量（支持Mask处理）"""
    
    def __init__(self, pinyin_vocab_size=500, pinyin_embed_dim=128, hidden_size=768):
        super().__init__()
        # 拼音字符嵌入（包括声母、韵母、声调），padding_idx=0 表示 [PAD] 的嵌入向量为全0
        self.pinyin_embedding = nn.Embedding(pinyin_vocab_size, pinyin_embed_dim, padding_idx=0)
        # 拼音序列编码（使用双向GRU）
        self.pinyin_gru = nn.GRU(pinyin_embed_dim, hidden_size // 2, 
                                  batch_first=True, bidirectional=True)
        # 投影到BERT隐藏层维度
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 增加LayerNorm稳定训练
        
    def forward(self, pinyin_ids, pinyin_lengths=None):
        """
        Args:
            pinyin_ids: [batch_size, seq_len, max_pinyin_len] 拼音字符ID序列
            pinyin_lengths: [batch_size, seq_len] 每个拼音的实际长度（可选，用于Mask）
        Returns:
            pinyin_features: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, pinyin_len = pinyin_ids.shape
        # 展平处理
        pinyin_ids_flat = pinyin_ids.view(-1, pinyin_len)  # [B*L, P]
        pinyin_embed = self.pinyin_embedding(pinyin_ids_flat)  # [B*L, P, E]
        
        # ⭐ 核心优化：使用 pack_padded_sequence 避免填充噪音
        if pinyin_lengths is not None:
            pinyin_lengths_flat = pinyin_lengths.view(-1).cpu()  # [B*L]
            # 避免长度为0（GRU要求长度>=1）
            pinyin_lengths_flat = torch.clamp(pinyin_lengths_flat, min=1)
            
            # 打包序列（只对有效部分编码）
            packed_embed = nn.utils.rnn.pack_padded_sequence(
                pinyin_embed, pinyin_lengths_flat, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.pinyin_gru(packed_embed)  # hidden: [2, B*L, H/2]
        else:
            # 如果没有长度信息，直接编码（会有轻微噪音，但不影响大局）
            _, hidden = self.pinyin_gru(pinyin_embed)  # hidden: [2, B*L, H/2]
        
        # 拼接双向GRU的隐藏状态
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [B*L, H]
        # 恢复形状并投影
        pinyin_features = hidden.view(batch_size, seq_len, -1)  # [B, L, H]
        pinyin_features = self.projection(pinyin_features)
        pinyin_features = self.layer_norm(pinyin_features)  # LayerNorm归一化
        return pinyin_features


class FusionLayer(nn.Module):
    """特征融合层：融合文本特征和拼音特征（支持多种融合策略）"""
    
    def __init__(self, hidden_size=768, fusion_type="gate", dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "gate":
            # ⭐ 改进的门控融合：增加非线性和Dropout
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),  # 使用Tanh增加非线性表达能力
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()  # 输出0-1之间的权重
            )
        elif fusion_type == "attention":
            # 注意力融合
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, 
                                                   dropout=dropout, batch_first=True)
        elif fusion_type == "bilinear":
            # ⭐ 新增：双线性融合（更强的交互能力）
            self.bilinear = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, pinyin_features, attention_mask=None):
        """
        Args:
            text_features: [B, L, H] MacBERT输出
            pinyin_features: [B, L, H] 拼音编码输出
            attention_mask: [B, L] 注意力掩码（可选）
        Returns:
            fused_features: [B, L, H] 融合后特征
        """
        if self.fusion_type == "gate":
            # 门控融合：动态学习两种特征的权重
            concat = torch.cat([text_features, pinyin_features], dim=-1)
            gate = self.gate(concat)  # [B, L, H]
            fused = gate * text_features + (1 - gate) * pinyin_features
        elif self.fusion_type == "attention":
            # 注意力融合：拼音特征作为Query，文本特征作为Key/Value
            fused, _ = self.attention(pinyin_features, text_features, text_features,
                                     key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
            fused = fused + text_features  # 残差连接
        elif self.fusion_type == "bilinear":
            # 双线性融合：捕捉两种特征的交互
            fused = self.bilinear(text_features, pinyin_features)
            fused = fused + text_features  # 残差连接
        else:
            # 简单加权（baseline）
            fused = 0.7 * text_features + 0.3 * pinyin_features
        
        # 应用Dropout和LayerNorm
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class Macbert4CSCWithPinyin(BertPreTrainedModel):
    """融合拼音特征的MacBERT中文拼写纠错模型"""
    
    def __init__(self, config, csc_config=None):
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name_or_path)
        super().__init__(bert_config)
        
        self.csc_config = csc_config or config
        self.hidden_size = bert_config.hidden_size
        
        # ⭐ 核心开关：是否使用拼音特征（用于消融实验）
        self.use_pinyin = getattr(self.csc_config, 'use_pinyin', True)  # 默认使用拼音
        
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
        """构建拼音字符词表（包含所有拼音字符和特殊符号）"""
        # 基础拼音字符：26个字母 + 5个声调（1-5）
        chars = list("abcdefghijklmnopqrstuvwxyz12345")
        self.pinyin_char2id = {"[PAD]": 0, "[UNK]": 1}
        for i, c in enumerate(chars):
            self.pinyin_char2id[c] = i + 2
        self.max_pinyin_len = 8  # 最长拼音如 "zhuang1"（7个字符），留1个buffer
        
    def text_to_pinyin_ids(self, texts, max_len):
        """
        将文本转换为拼音ID序列（同时返回长度信息用于Mask）
        Args:
            texts: 文本列表，如 ["我爱中国", "今天天气"]
            max_len: 序列最大长度（通常等于 input_ids.shape[1]）
        Returns:
            pinyin_ids: [batch_size, max_len, max_pinyin_len]
            pinyin_lengths: [batch_size, max_len] 每个拼音的实际长度
        """
        batch_pinyin_ids = []
        batch_pinyin_lengths = []
        
        for text in texts:
            # 获取每个字的拼音（带声调，如 "zhong1"）
            pinyins = pinyin(text, style=Style.TONE3, errors='ignore')
            seq_pinyin_ids = []
            seq_pinyin_lengths = []
            
            for py in pinyins:
                if py and py[0]:  # 确保拼音有效
                    py_str = py[0].lower()
                    # 将拼音字符串转为ID列表
                    char_ids = [self.pinyin_char2id.get(c, 1) for c in py_str]
                    actual_len = len(char_ids)  # ⭐ 记录实际长度（用于Mask）
                    # 截断或填充到固定长度
                    char_ids = char_ids[:self.max_pinyin_len]
                    char_ids += [0] * (self.max_pinyin_len - len(char_ids))
                else:
                    # 无效字符（如标点符号）用全0填充
                    char_ids = [0] * self.max_pinyin_len
                    actual_len = 0
                
                seq_pinyin_ids.append(char_ids)
                seq_pinyin_lengths.append(actual_len)
            
            # 填充序列长度到 max_len
            while len(seq_pinyin_ids) < max_len:
                seq_pinyin_ids.append([0] * self.max_pinyin_len)
                seq_pinyin_lengths.append(0)
            
            # 截断到 max_len
            seq_pinyin_ids = seq_pinyin_ids[:max_len]
            seq_pinyin_lengths = seq_pinyin_lengths[:max_len]
            
            batch_pinyin_ids.append(seq_pinyin_ids)
            batch_pinyin_lengths.append(seq_pinyin_lengths)
        
        return torch.tensor(batch_pinyin_ids), torch.tensor(batch_pinyin_lengths)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, pinyin_ids=None, pinyin_lengths=None, texts=None):
        """
        前向传播（支持动态拼音计算和Mask处理）
        Args:
            input_ids: [B, L] 输入token ID
            attention_mask: [B, L] 注意力掩码
            token_type_ids: [B, L] 句子类型ID
            labels: [B, L] 标签（正确的token ID）
            pinyin_ids: [B, L, P] 拼音ID序列（可选，建议预计算）
            pinyin_lengths: [B, L] 每个拼音的实际长度（可选，用于Mask）
            texts: List[str] 原始文本（仅在 pinyin_ids=None 时需要）
        """
        device = input_ids.device
        
        # 1. MacBERT编码（获取文本特征）
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
        if self.use_pinyin:
            # ========== 使用拼音特征（改进版）==========
            if pinyin_ids is None:
                # ⚠️ 动态计算拼音（会影响性能，建议在数据预处理时完成）
                if texts is None:
                    # 如果没有提供原始文本，则从 input_ids 反向解码
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.csc_config.pretrained_model_name_or_path)
                    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                # 将文本转换为拼音ID（同时获取长度信息）
                pinyin_ids, pinyin_lengths = self.text_to_pinyin_ids(texts, input_ids.shape[1])
                pinyin_ids = pinyin_ids.to(device)
                pinyin_lengths = pinyin_lengths.to(device)
            
            # 编码拼音特征（传入长度信息用于Mask）
            pinyin_features = self.pinyin_encoder(pinyin_ids, pinyin_lengths)  # [B, L, H]
            
            # 3. ⭐ 特征融合（核心创新）
            fused_features = self.fusion_layer(text_features, pinyin_features, attention_mask)  # [B, L, H]
        else:
            # ========== 不使用拼音特征（Baseline）==========
            fused_features = text_features  # 直接使用 BERT 特征，不融合拼音
        
        # 4. 错误检测（基于融合特征）
        det_logits = self.detect(fused_features).squeeze(-1)  # [B, L]
        if attention_mask is not None:
            det_probs = self.sigmoid(det_logits) * attention_mask
        else:
            det_probs = self.sigmoid(det_logits)
        
        # 5. 纠正预测（基于融合特征）
        cor_logits = self.correct(fused_features)  # [B, L, V]
        cor_probs = self.softmax(cor_logits)
        pred_prob, pred_ids = torch.max(cor_probs, dim=-1)
        if attention_mask is not None:
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
        
        if labels is not None:
            # 训练模式：计算损失
            det_labels = (input_ids != labels).float()
            det_loss = self.loss_bce(det_probs, det_labels)
            cor_loss = self.loss_ce(cor_logits.view(-1, cor_logits.size(-1)), labels.view(-1))
            
            # 总损失（可通过 csc_config.loss_det_rate 调节检测和纠正的权重）
            loss_det_rate = getattr(self.csc_config, 'loss_det_rate', 0.3)
            loss = (1 - loss_det_rate) * cor_loss + loss_det_rate * det_loss
            
            return (loss, det_loss, cor_loss, det_probs, cor_probs, pred_ids)
        else:
            # 推理模式：只返回预测结果
            return (cor_probs, pred_prob, pred_ids)