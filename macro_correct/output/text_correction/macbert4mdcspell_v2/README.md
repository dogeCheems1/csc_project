---
license: apache-2.0
language:
- zh
base_model:
- hfl/chinese-macbert-base
pipeline_tag: text-generation
tags:
- csc
- text-correct
- chinses-spelling-correct
- chinese-spelling-check
- 中文拼写纠错
- 文本纠错
- mdcspell
- macro-correct
---
# macbert4mdcspell
## 概述(macbert4mdcspell)
 - macro-correct, 中文拼写纠错CSC测评(文本纠错), 权重使用
 - 项目地址在 [https://github.com/yongzhuo/macro-correct](https://github.com/yongzhuo/macro-correct)
 - 本模型权重为macbert4mdcspell_v2, 使用mdcspell架构, 其特点是det_label和cor_label交互;
 - 训练时加入了macbert的mlm-loss, 推理时舍弃了macbert后面的部分;
 - 如何使用: 1.使用transformers调用; 2.使用[macro-correct](https://github.com/yongzhuo/macro-correct)项目调用; 详情见***三、调用(Usage)***;
 - 为了修复过纠问题, macbert4mdcspell_v2的MFT只70%的时间no-error-mask(0.15), 15%的时间target-to-target, 15%的时间不mask;

## 目录
* [一、测评(Test)](#一、测评(Test))
* [二、结论(Conclusion)](#二、结论(Conclusion))
* [三、调用(Usage)](#三、调用(Usage))
* [四、论文(Paper)](#四、论文(Paper))
* [五、参考(Refer)](#五、参考(Refer))
* [六、引用(Cite)](#六、引用(Cite))


## 一、测评(Test)
### 1.1 测评数据来源
地址为[Macropodus/csc_eval_public](https://huggingface.co/datasets/Macropodus/csc_eval_public), 所有训练数据均来自公网或开源数据, 训练数据为1千万左右, 混淆词典较大;
``` 
1.gen_de3.json(5545): '的地得'纠错, 由人民日报/学习强国/chinese-poetry等高质量数据人工生成;
2.lemon_v2.tet.json(1053): relm论文提出的数据, 多领域拼写纠错数据集(7个领域), ; 包括game(GAM), encyclopedia (ENC), contract (COT), medical care(MEC), car (CAR), novel (NOV), and news (NEW)等领域;
3.acc_rmrb.tet.json(4636): 来自NER-199801(人民日报高质量语料);
4.acc_xxqg.tet.json(5000): 来自学习强国网站的高质量语料;
5.gen_passage.tet.json(10000): 源数据为qwen生成的好词好句, 由几乎所有的开源数据汇总的混淆词典生成;
6.textproof.tet.json(1447): NLP竞赛数据, TextProofreadingCompetition;
7.gen_xxqg.tet.json(5000): 源数据为学习强国网站的高质量语料, 由几乎所有的开源数据汇总的混淆词典生成;
8.faspell.dev.json(1000): 视频字幕通过OCR后获取的数据集; 来自爱奇艺的论文faspell;
9.lomo_tet.json(5000): 主要为音似中文拼写纠错数据集; 来自腾讯; 人工标注的数据集CSCD-NS;
10.mcsc_tet.5000.json(5000): 医学拼写纠错; 来自腾讯医典APP的真实历史日志; 注意论文说该数据集只关注医学实体的纠错, 常用字等的纠错并不关注;
11.ecspell.dev.json(1500): 来自ECSpell论文, 包括(law/med/gov)等三个领域;
12.sighan2013.dev.json(1000): 来自sighan13会议;
13.sighan2014.dev.json(1062): 来自sighan14会议;
14.sighan2015.dev.json(1100): 来自sighan15会议;
```
### 1.2 测评数据预处理
```
测评数据都经过 全角转半角,繁简转化,标点符号标准化等操作;
```

### 1.3 其他说明
```
1.指标带common的极为宽松指标, 同开源项目pycorrector的评估指标;
2.指标带strict的极为严格指标, 同开源项目[wangwang110/CSC](https://github.com/wangwang110/CSC);
3.macbert4mdcspell_v1模型为训练使用mdcspell架构+bert的mlm-loss, 但是推理的时候只用bert-mlm;
4.acc_rmrb/acc_xxqg数据集没有错误, 用于评估模型的误纠率(过度纠错);
5.qwen25_1-5b_pycorrector的模型为shibing624/chinese-text-correction-1.5b, 其训练数据包括了lemon_v2/mcsc_tet/ecspell的验证集和测试集, 其他的bert类模型的训练不包括验证集和测试集;
```


## 二、重要指标
### 2.1 F1(common_cor_f1)
| model/common_cor_f1     | avg| gen_de3| lemon_v2| gen_passage| text_proof| gen_xxqg| faspell| lomo_tet| mcsc_tet| ecspell| sighan2013| sighan2014| sighan2015 |
|:------------------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector | 45.8| 42.44| 42.89| 31.49| 46.31| 26.06| 32.7| 44.83| 27.93| 55.51| 70.89| 61.72| 66.81 |
| qwen25_1-5b_pycorrector | 45.11| 27.29| 89.48| 14.61| 83.9| 13.84| 18.2| 36.71| 96.29| 88.2| 36.41| 15.64| 20.73 |
| bert4csc_v1             | 62.28| 93.73| 61.99| 44.79| 68.0| 35.03| 48.28| 61.8| 64.41| 79.11| 77.66| 51.01| 61.54 |
| macbert4csc_v1          | 68.55| 96.67| 65.63| 48.4| 75.65| 38.43| 51.76| 70.11| 80.63| 85.55| 81.38| 57.63| 70.7 |
| macbert4csc_v2          | 68.6| 96.74| 66.02| 48.26| 75.78| 38.84| 51.91| 70.17| 80.71| 85.61| 80.97| 58.22| 69.95 |
| macbert4mdcspell_v1     | 71.1| 96.42| 70.06| 52.55| 79.61| 43.37| 53.85| 70.9| 82.38| 87.46| 84.2| 61.08| 71.32 |
| macbert4mdcspell_v2     | 71.23| 96.42| 65.8| 52.35| 75.94| 43.5| 53.82| 72.66| 82.28| 88.69| 82.51| 65.59| 75.26 |

### 2.2 acc(common_cor_acc)
| model/common_cor_acc| avg| gen_de3| lemon_v2| gen_passage| text_proof| gen_xxqg| faspell| lomo_tet| mcsc_tet| ecspell| sighan2013| sighan2014| sighan2015 |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector| 48.26| 26.96| 28.68| 34.16| 55.29| 28.38| 22.2| 60.96| 57.16| 67.73| 55.9| 68.93| 72.73 |
| qwen25_1-5b_pycorrector| 46.09| 15.82| 81.29| 22.96| 82.17| 19.04| 12.8| 50.2| 96.4| 89.13| 22.8| 27.87| 32.55 |
| bert4csc_v1| 60.76| 88.21| 45.96| 43.13| 68.97| 35.0| 34.0| 65.86| 73.26| 81.8| 64.5| 61.11| 67.27 |
| macbert4csc_v1| 65.34| 93.56| 49.76| 44.98| 74.64| 36.1| 37.0| 73.0| 83.6| 86.87| 69.2| 62.62| 72.73 |
| macbert4csc_v2| 65.22| 93.69| 50.14| 44.92| 74.64| 36.26| 37.0| 72.72| 83.66| 86.93| 68.5| 62.43| 71.73 |
| macbert4mdcspell_v1| 67.15| 93.09| 54.8| 47.71| 78.09| 39.52| 38.8| 71.92| 84.78| 88.27| 73.2| 63.28| 72.36 |
| macbert4mdcspell_v2     | 68.31| 93.09| 50.05| 48.72| 75.74| 40.52| 38.9| 76.9| 84.8| 89.73| 71.0| 71.94| 78.36 |

### 2.3 acc(acc_true, thr=0.75)
| model/acc               | avg| acc_rmrb| acc_xxqg |
|:------------------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector | 99.24| 99.22| 99.26 |
| qwen25_1-5b_pycorrector | 82.0| 77.14| 86.86 |
| bert4csc_v1             | 98.71| 98.36| 99.06 |
| macbert4csc_v1          | 97.72| 96.72| 98.72 |
| macbert4csc_v2          | 97.89| 96.98| 98.8 |
| macbert4mdcspell_v1     | 97.75| 96.51| 98.98 |
| macbert4mdcspell_v2     | 99.54| 99.22| 99.86 |

## 二、结论(Conclusion)
```
1.macbert4csc_v1/macbert4csc_v2/macbert4mdcspell_v1等模型使用多种领域数据训练, 比较均衡, 也适合作为第一步的预训练模型, 可用于专有领域数据的继续微调;
2.比较macbert4csc_pycorrector/bertbase4csc_v1/macbert4csc_v2/macbert4mdcspell_v1, 观察表2.3, 可以发现训练数据越多, 准确率提升的同时, 误纠率也会稍微高一些;
3.MFT(Mask-Correct)依旧有效, 不过对于数据量足够的情形提升不明显, 可能也是误纠率升高的一个重要原因;
4.训练数据中也存在文言文数据, 训练好的模型也支持文言文纠错;
5.训练好的模型对"地得的"等高频错误具有较高的识别率和纠错率;
6.macbert4mdcspell_v2的MFT只70%的时间no-error-mask(0.15), 15%的时间target-to-target, 15%的时间不mask;
```

## 三、调用(Usage)
### 3.1 使用macro-correct
```
import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"
from macro_correct import correct
### 默认纠错(list输入)
text_list = ["真麻烦你了。希望你们好好的跳无",
             "少先队员因该为老人让坐",
             "机七学习是人工智能领遇最能体现智能的一个分知",
             "一只小鱼船浮在平净的河面上"
             ]
text_csc = correct(text_list)
print("默认纠错(list输入):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)

"""
默认纠错(list输入):
{'index': 0, 'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12, 0.6584], ['无', '舞', 14, 1.0]]}
{'index': 1, 'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4, 0.995]]}
{'index': 2, 'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1, 0.9998], ['遇', '域', 10, 0.9999], ['知', '支', 21, 1.0]]}
{'index': 3, 'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8, 0.9961]]}
"""
```

### 3.2 使用 transformers
```
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: transformers直接加载bert类模型测试


import traceback
import time
import sys
import os
os.environ["USE_TORCH"] = "1"
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch

# pretrained_model_name_or_path = "shibing624/macbert4csc-base-chinese"
pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v2"
# pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v2"
# pretrained_model_name_or_path = "Macropodus/bert4csc_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128

print("load model, please wait a few minute!")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
model.to(device)
print("load model success!")

texts = [
    "机七学习是人工智能领遇最能体现智能的一个分知",
    "我是练习时长两念半的鸽仁练习生蔡徐坤",
    "真麻烦你了。希望你们好好的跳无",
    "他法语说的很好，的语也不错",
    "遇到一位很棒的奴生跟我疗天",
    "我们为这个目标努力不解",
]
len_mid = min(max_len, max([len(t)+2 for t in texts]))

with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, max_length=len_mid,
                                return_tensors="pt").to(device))

def get_errors(source, target):
    """   极简方法获取 errors   """
    len_min = min(len(source), len(target))
    errors = []
    for idx in range(len_min):
        if source[idx] != target[idx]:
            errors.append([source[idx], target[idx], idx])
    return errors

result = []
for probs, source in zip(outputs.logits, texts):
    ids = torch.argmax(probs, dim=-1)
    tokens_space = tokenizer.decode(ids[1:-1], skip_special_tokens=False)
    text_new = tokens_space.replace(" ", "")
    target = text_new[:len(source)]
    errors = get_errors(source, target)
    print(source, " => ", target, errors)
    result.append([target, errors])
print(result)
"""
机七学习是人工智能领遇最能体现智能的一个分知  =>  机器学习是人工智能领域最能体现智能的一个分支 [['七', '器', 1], ['遇', '域', 10], ['知', '支', 21]]
我是练习时长两念半的鸽仁练习生蔡徐坤  =>  我是练习时长两年半的个人练习生蔡徐坤 [['念', '年', 7], ['鸽', '个', 10], ['仁', '人', 11]]
真麻烦你了。希望你们好好的跳无  =>  真麻烦你了。希望你们好好地跳舞 [['的', '地', 12], ['无', '舞', 14]]
他法语说的很好，的语也不错  =>  他法语说得很好，德语也不错 [['的', '得', 4], ['的', '德', 8]]
遇到一位很棒的奴生跟我疗天  =>  遇到一位很棒的女生跟我聊天 [['奴', '女', 7], ['疗', '聊', 11]]
我们为这个目标努力不解  =>  我们为这个目标努力不懈 [['解', '懈', 10]]
"""
```

## 四、论文(Paper)
 - 2024-Refining: [Refining Corpora from a Model Calibration Perspective for Chinese](https://arxiv.org/abs/2407.15498)
 - 2024-ReLM: [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796)
 - 2024-DICS: [DISC: Plug-and-Play Decoding Intervention with Similarity of Characters for Chinese Spelling Check](https://arxiv.org/abs/2412.12863)

 - 2023-Bi-DCSpell: [A Bi-directional Detector-Corrector Interactive Framework for Chinese Spelling Check]()
 - 2023-BERT-MFT: [Rethinking Masked Language Modeling for Chinese Spelling Correction](https://arxiv.org/abs/2305.17721)
 - 2023-PTCSpell: [PTCSpell: Pre-trained Corrector Based on Character Shape and Pinyin for Chinese Spelling Correction](https://arxiv.org/abs/2212.04068)
 - 2023-DR-CSC: [A Frustratingly Easy Plug-and-Play Detection-and-Reasoning Module for Chinese](https://aclanthology.org/2023.findings-emnlp.771)
 - 2023-DROM: [Disentangled Phonetic Representation for Chinese Spelling Correction](https://arxiv.org/abs/2305.14783)
 - 2023-EGCM: [An Error-Guided Correction Model for Chinese Spelling Error Correction](https://arxiv.org/abs/2301.06323)
 - 2023-IGPI: [Investigating Glyph-Phonetic Information for Chinese Spell Checking: What Works and What’s Next?](https://arxiv.org/abs/2212.04068)
 - 2023-CL: [Contextual Similarity is More Valuable than Character Similarity-An Empirical Study for Chinese Spell Checking]()

 - 2022-CRASpell: [CRASpell: A Contextual Typo Robust Approach to Improve Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.237)
 - 2022-MDCSpell: [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98)
 - 2022-SCOPE: [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/abs/2210.10996)
 - 2022-ECOPO: [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/abs/2203.00991)

 - 2021-MLMPhonetics: [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198)
 - 2021-ChineseBERT: [ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://aclanthology.org/2021.acl-long.161/)
 - 2021-BERTCrsGad: [Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122)
 - 2021-ThinkTwice: [Think Twice: A Post-Processing Approach for the Chinese Spelling Error Correction](https://www.mdpi.com/2076-3417/11/13/5832)
 - 2021-PHMOSpell: [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Chec](https://aclanthology.org/2021.acl-long.464)
 - 2021-SpellBERT: [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287)
 - 2021-TwoWays: [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56)
 - 2021-ReaLiSe: [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://arxiv.org/abs/2105.12306)
 - 2021-DCSpell: [DCSpell: A Detector-Corrector Framework for Chinese Spelling Error Correction](https://dl.acm.org/doi/10.1145/3404835.3463050)
 - 2021-PLOME: [PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction](https://aclanthology.org/2021.acl-long.233)
 - 2021-DCN: [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216/)

 - 2020-SoftMaskBERT: [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
 - 2020-SpellGCN: [SpellGCN：Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/abs/2004.14166)
 - 2020-ChunkCSC: [Chunk-based Chinese Spelling Check with Global Optimization](https://aclanthology.org/2020.findings-emnlp.184)
 - 2020-MacBERT: [Revisiting Pre-Trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922)

 - 2019-FASPell: [FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm](https://aclanthology.org/D19-5522)
 - 2018-Hybrid: [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Checking](https://aclanthology.org/D18-1273)

 - 2015-Sighan15: [Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check](https://aclanthology.org/W15-3106/)
 - 2014-Sighan14: [Overview of SIGHAN 2014 Bake-off for Chinese Spelling Check](https://aclanthology.org/W14-6820/)
 - 2013-Sighan13: [Chinese Spelling Check Evaluation at SIGHAN Bake-off 2013](https://aclanthology.org/W13-4406/)

## 五、参考(Refer)
 - [nghuyong/Chinese-text-correction-papers](https://github.com/nghuyong/Chinese-text-correction-papers)
 - [destwang/CTCResources](https://github.com/destwang/CTCResources)
 - [wangwang110/CSC](https://github.com/wangwang110/CSC)
 - [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
 - [chinese-poetry/huajianji](https://github.com/chinese-poetry/huajianji)
 - [garychowcmu/daizhigev20](https://github.com/garychowcmu/daizhigev20)
 - [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [Macropodus/xuexiqiangguo_428w](https://huggingface.co/datasets/Macropodus/xuexiqiangguo_428w)
 - [Macropodus/csc_clean_wang271k](https://huggingface.co/datasets/Macropodus/csc_clean_wang271k)
 - [Macropodus/csc_eval_public](https://huggingface.co/datasets//Macropodus/csc_eval_public)
 - [shibing624/pycorrector](https://github.com/shibing624/pycorrector)
 - [iioSnail/MDCSpell_pytorch](https://github.com/iioSnail/MDCSpell_pytorch)
 - [gingasan/lemon](https://github.com/gingasan/lemon)
 - [Claude-Liu/ReLM](https://github.com/Claude-Liu/ReLM)


## 六、引用(Cite)
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@software{macro-correct,
    url = {https://github.com/yongzhuo/macro-correct},
    author = {Yongzhuo Mo},
    title = {macro-correct},
    year = {2025}
```