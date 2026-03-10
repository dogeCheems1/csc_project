# !/usr/bin/python
# -*- coding: utf-8 -*-
"""MacBERT with pinyin encoder and fusion layer for CSC."""

import torch
import torch.nn as nn
from pypinyin import Style, pinyin
from transformers import BertConfig, BertForMaskedLM, BertPreTrainedModel


class PinyinEncoder(nn.Module):
    def __init__(self, pinyin_vocab_size=500, pinyin_embed_dim=128, hidden_size=768):
        super().__init__()
        self.pinyin_embedding = nn.Embedding(pinyin_vocab_size, pinyin_embed_dim, padding_idx=0)
        self.pinyin_gru = nn.GRU(
            pinyin_embed_dim,
            hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pinyin_ids, pinyin_lengths=None):
        batch_size, seq_len, pinyin_len = pinyin_ids.shape
        pinyin_ids_flat = pinyin_ids.view(-1, pinyin_len)
        pinyin_embed = self.pinyin_embedding(pinyin_ids_flat)

        if pinyin_lengths is not None:
            pinyin_lengths_flat = pinyin_lengths.view(-1).cpu()
            pinyin_lengths_flat = torch.clamp(pinyin_lengths_flat, min=1)
            packed_embed = nn.utils.rnn.pack_padded_sequence(
                pinyin_embed,
                pinyin_lengths_flat,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.pinyin_gru(packed_embed)
        else:
            _, hidden = self.pinyin_gru(pinyin_embed)

        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        pinyin_features = hidden.view(batch_size, seq_len, -1)
        pinyin_features = self.projection(pinyin_features)
        pinyin_features = self.layer_norm(pinyin_features)
        return pinyin_features


class FusionLayer(nn.Module):
    def __init__(self, hidden_size=768, fusion_type="gate", dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        self.last_gate_stats = None
        if fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features, pinyin_features, attention_mask=None):
        self.last_gate_stats = None
        if self.fusion_type == "gate":
            concat = torch.cat([text_features, pinyin_features], dim=-1)
            gate = self.gate(concat)
            fused = gate * text_features + (1.0 - gate) * pinyin_features
            self.last_gate_stats = {"mean": gate.mean().item(), "std": gate.std().item()}
        elif self.fusion_type == "attention":
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            fused, _ = self.attention(pinyin_features, text_features, text_features, key_padding_mask=key_padding_mask)
            fused = fused + text_features
        elif self.fusion_type == "bilinear":
            fused = self.bilinear(text_features, pinyin_features) + text_features
        else:
            fused = 0.7 * text_features + 0.3 * pinyin_features

        fused = self.dropout(fused)
        return self.layer_norm(fused)


class Macbert4CSCWithPinyin(BertPreTrainedModel):
    def __init__(self, config, csc_config=None):
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name_or_path)
        super().__init__(bert_config)
        self.csc_config = csc_config or config
        self.hidden_size = bert_config.hidden_size
        self.use_pinyin = getattr(self.csc_config, "use_pinyin", True)
        self.last_gate_stats = None

        if getattr(self.csc_config, "flag_train", True):
            self.bert = BertForMaskedLM.from_pretrained(
                config.pretrained_model_name_or_path,
                output_hidden_states=True,
            )
        else:
            self.bert = BertForMaskedLM(bert_config)

        self.pinyin_encoder = PinyinEncoder(
            pinyin_vocab_size=getattr(self.csc_config, "pinyin_vocab_size", 500),
            pinyin_embed_dim=getattr(self.csc_config, "pinyin_embed_dim", 128),
            hidden_size=self.hidden_size,
        )
        self.fusion_layer = FusionLayer(
            hidden_size=self.hidden_size,
            fusion_type=getattr(self.csc_config, "fusion_type", "gate"),
        )

        self.detect = nn.Linear(self.hidden_size, 1)
        self.correct = nn.Linear(self.hidden_size, bert_config.vocab_size)
        self.correct.weight.data = self.bert.bert.embeddings.word_embeddings.weight.data

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_bce = nn.BCELoss()
        self._build_pinyin_vocab()

    def _build_pinyin_vocab(self):
        chars = list("abcdefghijklmnopqrstuvwxyz12345")
        self.pinyin_char2id = {"[PAD]": 0, "[UNK]": 1}
        for i, c in enumerate(chars):
            self.pinyin_char2id[c] = i + 2
        self.max_pinyin_len = 8

    def _encode_single_pinyin(self, py_str):
        if not py_str:
            return [0] * self.max_pinyin_len, 0
        char_ids = [self.pinyin_char2id.get(ch, 1) for ch in py_str.lower()]
        actual_len = min(len(char_ids), self.max_pinyin_len)
        char_ids = char_ids[: self.max_pinyin_len]
        char_ids += [0] * (self.max_pinyin_len - len(char_ids))
        return char_ids, actual_len

    def _text_chars_with_ids_to_pinyin(self, text_chars, input_ids):
        if not isinstance(text_chars, list):
            text_chars = list(text_chars)
        pinyins = pinyin(text_chars, style=Style.TONE3, errors=lambda x: [[""] for _ in x])
        py_index = 0
        seq_pinyin_ids = []
        seq_pinyin_lengths = []
        for token_id in input_ids:
            if token_id == 0 or token_id in (101, 102):  # [PAD]/[CLS]/[SEP]
                seq_pinyin_ids.append([0] * self.max_pinyin_len)
                seq_pinyin_lengths.append(0)
                continue
            py_str = pinyins[py_index][0] if py_index < len(pinyins) and pinyins[py_index] else ""
            char_ids, actual_len = self._encode_single_pinyin(py_str)
            seq_pinyin_ids.append(char_ids)
            seq_pinyin_lengths.append(actual_len)
            py_index += 1
        return seq_pinyin_ids, seq_pinyin_lengths

    def text_to_pinyin_ids(self, text_or_chars, input_ids_or_maxlen):
        """
        Backward compatible API:
        1) text_to_pinyin_ids(text_chars, input_ids) -> aligned single sample list/list
        2) text_to_pinyin_ids(texts, max_len) -> batch tensors (legacy helper usage)
        """
        if isinstance(input_ids_or_maxlen, int):
            max_len = input_ids_or_maxlen
            texts = text_or_chars if isinstance(text_or_chars, list) else [text_or_chars]
            batch_pinyin_ids = []
            batch_pinyin_lengths = []
            for text in texts:
                chars = list(text)
                seq_ids = []
                seq_lens = []
                for py in pinyin(chars, style=Style.TONE3, errors=lambda x: [[""] for _ in x]):
                    py_str = py[0] if py else ""
                    char_ids, actual_len = self._encode_single_pinyin(py_str)
                    seq_ids.append(char_ids)
                    seq_lens.append(actual_len)
                seq_ids = seq_ids[:max_len] + [[0] * self.max_pinyin_len] * max(0, max_len - len(seq_ids))
                seq_lens = seq_lens[:max_len] + [0] * max(0, max_len - len(seq_lens))
                batch_pinyin_ids.append(seq_ids[:max_len])
                batch_pinyin_lengths.append(seq_lens[:max_len])
            return torch.tensor(batch_pinyin_ids), torch.tensor(batch_pinyin_lengths)

        return self._text_chars_with_ids_to_pinyin(text_or_chars, input_ids_or_maxlen)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        pinyin_ids=None,
        pinyin_lengths=None,
        texts=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        text_features = outputs.hidden_states[-1]

        if self.use_pinyin:
            if pinyin_ids is None:
                raise ValueError("pinyin_ids is required when use_pinyin=True")
            pinyin_features = self.pinyin_encoder(pinyin_ids, pinyin_lengths)
            fused_features = self.fusion_layer(text_features, pinyin_features, attention_mask)
            self.last_gate_stats = self.fusion_layer.last_gate_stats
        else:
            fused_features = text_features
            self.last_gate_stats = None

        det_logits = self.detect(fused_features).squeeze(-1)
        if attention_mask is not None:
            det_probs = self.sigmoid(det_logits) * attention_mask
        else:
            det_probs = self.sigmoid(det_logits)

        cor_logits = self.correct(fused_features)
        cor_probs = self.softmax(cor_logits)
        pred_prob, pred_ids = torch.max(cor_probs, dim=-1)
        if attention_mask is not None:
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)

        if labels is not None:
            det_labels = (input_ids != labels).float()
            det_loss = self.loss_bce(det_probs, det_labels)
            cor_loss = self.loss_ce(cor_logits.view(-1, cor_logits.size(-1)), labels.view(-1))
            loss_det_rate = getattr(self.csc_config, "loss_det_rate", 0.3)
            loss = (1 - loss_det_rate) * cor_loss + loss_det_rate * det_loss
            return (loss, det_loss, cor_loss, det_probs, cor_probs, pred_ids)
        return (cor_probs, pred_prob, pred_ids)
